"""
Real-time LFM2-Audio WebSocket API with streaming audio and interruption support.
Designed for web-based voice calls with minimal latency.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, "/Users/hamishfromatech/liquid-audio/src")

from liquid_audio import ChatState, LFMModality
from liquid_audio.demo.model import lfm2_audio, mimi, proc, device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LFM2-Audio Realtime API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audio configuration
SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80ms at 24kHz
CHUNK_DURATION_MS = int((FRAME_SIZE / SAMPLE_RATE) * 1000)


@dataclass
class AudioBuffer:
    """Manages incoming audio chunks."""
    buffer: deque = None
    lock: asyncio.Lock = None
    
    def __post_init__(self):
        if self.buffer is None:
            self.buffer = deque(maxlen=100)
        if self.lock is None:
            self.lock = asyncio.Lock()
    
    async def add_chunk(self, chunk: np.ndarray):
        """Add audio chunk to buffer."""
        async with self.lock:
            self.buffer.append(chunk)
    
    async def get_chunk(self) -> Optional[np.ndarray]:
        """Get next audio chunk from buffer."""
        async with self.lock:
            if self.buffer:
                return self.buffer.popleft()
        return None
    
    async def has_data(self) -> bool:
        """Check if buffer has data."""
        async with self.lock:
            return len(self.buffer) > 0


def _sanitize_temperature(value: float | int | None) -> float:
    try:
        temp = float(value)
    except (TypeError, ValueError):
        temp = 1.0
    return temp if temp > 0 else 0.7


def _sanitize_topk(value: float | int | None) -> int | None:
    if value in (None, 0):
        return None
    try:
        topk = int(value)
    except (TypeError, ValueError):
        return 4
    return max(1, topk)


async def _send_json_safe(websocket: WebSocket, payload: dict) -> bool:
    """Attempt to send JSON over WebSocket, return False if disconnected."""
    if websocket.application_state is WebSocketState.DISCONNECTED or websocket.client_state is WebSocketState.DISCONNECTED:
        return False
    try:
        await websocket.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError:
        # Happens if close frame already sent
        return False


class RealtimeConversation:
    """Manages real-time conversation with streaming audio."""
    
    def __init__(self, system_prompt: str = ""):
        self.chat = ChatState(proc)
        self.system_prompt = system_prompt or "You are a helpful, friendly assistant. Keep responses concise and natural."
        self.audio_buffer = AudioBuffer()
        self.is_generating = False
        self.should_interrupt = False
        self.conversation_history = deque(maxlen=5)
        self._initialize_chat()
    
    def _initialize_chat(self):
        """Initialize chat with system prompt."""
        self.chat.new_turn("system")
        self.chat.add_text(self.system_prompt)
        self.chat.end_turn()
        self.chat.new_turn("user")
    
    async def add_audio_chunk(self, chunk: np.ndarray):
        """Add audio chunk to buffer."""
        await self.audio_buffer.add_chunk(chunk)
    
    def interrupt_generation(self):
        """Signal to interrupt current generation."""
        self.should_interrupt = True
    
    async def process_audio_stream(self, websocket: WebSocket, temp: float | int | None = 1.0, topk: float | int | None = 4):
        temperature = _sanitize_temperature(temp)
        top_k = _sanitize_topk(topk)
        """Process streaming audio and generate responses."""
        try:
            # Collect audio until we have enough for processing
            audio_chunks = []
            min_chunks = 3  # Collect at least 3 chunks before processing
            
            while not self.should_interrupt:
                chunk = await self.audio_buffer.get_chunk()
                if chunk is not None:
                    audio_chunks.append(chunk)
                    
                    # Once we have enough chunks, start processing
                    if len(audio_chunks) >= min_chunks:
                        await self._generate_response(websocket, audio_chunks, temperature, top_k)
                        audio_chunks = []
                        min_chunks = 1  # After first response, process more frequently
                else:
                    await asyncio.sleep(0.01)
            
            # Process remaining audio
            if audio_chunks:
                await self._generate_response(websocket, audio_chunks, temperature, top_k)
        
        except Exception as e:
            logger.error(f"Error processing audio stream: {e}", exc_info=True)
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    async def _generate_response(self, websocket: WebSocket, audio_chunks: list, temp: float, topk: int | None):
        """Generate response from audio chunks."""
        self.is_generating = True
        self.should_interrupt = False
        
        try:
            # Concatenate audio chunks
            audio_data = np.concatenate(audio_chunks)
            
            # Add audio to chat
            audio_tensor = torch.tensor(audio_data.astype(np.float32) / 32_768.0, dtype=torch.float).unsqueeze(0)
            self.chat.add_audio(audio_tensor, SAMPLE_RATE)
            self.chat.end_turn()
            
            # Start generation
            self.chat.new_turn("assistant")
            
            text_response = ""
            audio_response = []
            
            with torch.no_grad(), mimi.streaming(1):
                for token in lfm2_audio.generate_interleaved(
                    **self.chat,
                    max_new_tokens=512,
                    audio_temperature=temp,
                    audio_top_k=topk,
                ):
                    if self.should_interrupt:
                        logger.info("Generation interrupted")
                        break
                    
                    # Handle text tokens
                    if token.numel() == 1:
                        text = proc.text.decode(token)
                        text_response += text
                        ok = await _send_json_safe(
                            websocket,
                            {
                                "type": "text_chunk",
                                "text": text,
                            },
                        )
                        if not ok:
                            logger.info("Client disconnected during text streaming")
                            break
                    
                    # Handle audio tokens
                    elif token.numel() == 8:
                        # Decode audio frame
                        wav_chunk = mimi.decode(token.unsqueeze(0).unsqueeze(-1))[0]
                        audio_response.append(wav_chunk)
                        
                        # Send audio chunk (convert to int16 and hex)
                        audio_int16 = (wav_chunk.cpu().numpy() * 32767).astype(np.int16)
                        audio_hex = audio_int16.tobytes().hex()
                        
                        ok = await _send_json_safe(
                            websocket,
                            {
                                "type": "audio_chunk",
                                "audio": audio_hex,
                                "duration_ms": CHUNK_DURATION_MS,
                            },
                        )
                        if not ok:
                            logger.info("Client disconnected during audio streaming")
                            break
            
            # Finalize response
            self.chat.end_turn()
            self.chat.new_turn("user")
            
            # Update history
            self.conversation_history.append({"role": "user", "text": "[Audio message]"})
            if text_response:
                self.conversation_history.append({"role": "assistant", "text": text_response.removesuffix("<|text_end|>").strip()})
            
            # Send completion signal
            await _send_json_safe(
                websocket,
                {
                    "type": "response_complete",
                    "text": text_response.removesuffix("<|text_end|>").strip(),
                },
            )
        
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        
        finally:
            self.is_generating = False


@app.websocket("/ws/call")
async def websocket_call(websocket: WebSocket):
    """Real-time voice call WebSocket endpoint."""
    await websocket.accept()
    
    conversation = None
    audio_task = None
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            # Initialize conversation
            if message_type == "init":
                system_prompt = data.get("system_prompt", "")
                conversation = RealtimeConversation(system_prompt)
                
                # Start audio processing task
                audio_task = asyncio.create_task(
                    conversation.process_audio_stream(
                        websocket,
                        temp=data.get("temperature", 1.0),
                        topk=data.get("top_k", 4)
                    )
                )
                
                await _send_json_safe(
                    websocket,
                    {
                        "type": "init_response",
                        "status": "ready",
                    },
                )
                logger.info("Conversation initialized")
            
            # Handle audio chunks
            elif message_type == "audio_chunk":
                if not conversation:
                    await websocket.send_json({"type": "error", "message": "Conversation not initialized"})
                    continue
                
                try:
                    # Decode audio from hex
                    audio_hex = data.get("audio", "")
                    if not audio_hex:
                        continue
                    
                    audio_bytes = bytes.fromhex(audio_hex)
                    
                    # Skip WAV header if present
                    if len(audio_bytes) > 44 and audio_bytes[:4] == b'RIFF':
                        # Find the data chunk
                        try:
                            data_pos = audio_bytes.find(b'data')
                            if data_pos > 0:
                                # Skip 'data' + 4-byte size
                                audio_data_start = data_pos + 8
                                audio_bytes = audio_bytes[audio_data_start:]
                        except:
                            # If parsing fails, skip first 44 bytes
                            audio_bytes = audio_bytes[44:]
                    
                    # Ensure buffer size is even (required for int16)
                    if len(audio_bytes) % 2 != 0:
                        audio_bytes = audio_bytes[:-1]
                    
                    # Convert to int16 array
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    if len(audio_data) > 0:
                        # Add to buffer
                        await conversation.add_audio_chunk(audio_data)
                
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    await _send_json_safe(
                        websocket,
                        {
                            "type": "error",
                            "message": f"Invalid audio data: {str(e)}",
                        },
                    )
            
            # Handle interruption
            elif message_type == "interrupt":
                if conversation:
                    conversation.interrupt_generation()
                    await _send_json_safe(
                        websocket,
                        {
                            "type": "interrupted",
                            "status": "generation stopped",
                        },
                    )
            
            # Handle reset
            elif message_type == "reset":
                if audio_task:
                    audio_task.cancel()
                    try:
                        await audio_task
                    except asyncio.CancelledError:
                        pass
                
                conversation = RealtimeConversation(data.get("system_prompt", ""))
                
                audio_task = asyncio.create_task(
                    conversation.process_audio_stream(
                        websocket,
                        temp=data.get("temperature", 1.0),
                        topk=data.get("top_k", 4)
                    )
                )
                
                await _send_json_safe(
                    websocket,
                    {
                        "type": "reset_response",
                        "status": "reset",
                    },
                )
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        if audio_task:
            audio_task.cancel()
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model": "LFM2-Audio"}


@app.get("/info")
async def model_info():
    """Get model information."""
    return {
        "model": "LFM2-Audio-1.5B",
        "capabilities": ["real-time-speech-to-speech", "streaming", "interruption"],
        "sample_rate": SAMPLE_RATE,
        "frame_size": FRAME_SIZE,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8493)
