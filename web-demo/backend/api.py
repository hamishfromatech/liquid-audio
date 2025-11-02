"""
FastAPI backend for LFM2-Audio voice call interface.
Provides WebSocket and REST endpoints for real-time speech-to-speech conversations.
"""

import asyncio
import json
import logging
from collections import deque
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import model components
import sys
sys.path.insert(0, "/Users/hamishfromatech/liquid-audio/src")

from liquid_audio import ChatState, LFMModality
from liquid_audio.demo.model import lfm2_audio, mimi, proc, device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LFM2-Audio API", version="1.0.0")

# Add CORS middleware for HTTP requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: WebSocket connections don't use CORS headers, but the middleware helps with preflight requests


class ConversationState:
    """Manages conversation state for a WebSocket connection."""
    
    def __init__(self, system_prompt: str = ""):
        self.chat = ChatState(proc)
        self.system_prompt = system_prompt or "Respond with interleaved text and audio. You are a helpful, friendly assistant. Keep responses concise and natural."
        self.conversation_history: deque = deque(maxlen=5)
        self._initialize_chat()
    
    def _initialize_chat(self):
        """Initialize chat with system prompt."""
        self.chat.new_turn("system")
        self.chat.add_text(self.system_prompt)
        self.chat.end_turn()
        self.chat.new_turn("user")
    
    def add_audio_input(self, audio_data: np.ndarray, sample_rate: int = 24000):
        """Add audio input to the conversation."""
        # Normalize audio to float and reshape to [1, num_samples] (batch_size=1)
        audio_float = audio_data.astype(np.float32) / 32_768.0
        audio_tensor = torch.tensor(audio_float, dtype=torch.float).unsqueeze(0)  # Add batch dimension
        self.chat.add_audio(audio_tensor, sample_rate)
        self.chat.end_turn()
    
    def add_text_input(self, text: str):
        """Add text input to the conversation."""
        self.chat.add_text(text)
        self.chat.end_turn()
    
    def start_assistant_turn(self):
        """Start the assistant's turn."""
        self.chat.new_turn("assistant")
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})


def generate_response(
    chat: ChatState,
    temp: Optional[float] = 1.0,
    topk: Optional[int] = 4,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[LFMModality], str]:
    """Generate response from the model."""
    
    if temp == 0:
        temp = None
    if topk == 0:
        topk = None
    
    if temp is not None:
        temp = float(temp)
    if topk is not None:
        topk = int(topk)
    
    out_text: list[torch.Tensor] = []
    out_audio: list[torch.Tensor] = []
    out_modality: list[LFMModality] = []
    assistant_text = ""
    
    q: Queue[torch.Tensor | None] = Queue()
    
    def producer():
        with torch.no_grad(), mimi.streaming(1):
            for t in lfm2_audio.generate_interleaved(
                **chat,
                max_new_tokens=1024,
                audio_temperature=temp,
                audio_top_k=topk,
            ):
                q.put(t)
                
                if t.numel() > 1:
                    if (t == 2048).any():
                        continue
                    
                    wav_chunk = mimi.decode(t.unsqueeze(0).unsqueeze(-1))[0]
                    q.put(wav_chunk)
        
        q.put(None)
    
    producer_thread = Thread(target=producer, daemon=True)
    producer_thread.start()
    
    while True:
        t = q.get()
        if t is None:
            break
        elif t.numel() == 1:  # text
            out_text.append(t)
            out_modality.append(LFMModality.TEXT)
            token_text = proc.text.decode(t)
            assistant_text += token_text
        elif t.numel() == 8:
            out_audio.append(t)
            out_modality.append(LFMModality.AUDIO_OUT)
        elif t.numel() == 1920:
            # Audio chunk
            pass
        else:
            logger.warning(f"Unexpected tensor shape: {t.shape}")
    
    producer_thread.join(timeout=5)
    
    return out_text, out_audio, out_modality, assistant_text.removesuffix("<|text_end|>").strip()


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time voice chat."""
    await websocket.accept()
    
    conversation = None
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            # Initialize conversation
            if message_type == "init":
                system_prompt = data.get("system_prompt", "")
                conversation = ConversationState(system_prompt)
                await websocket.send_json({
                    "type": "init_response",
                    "status": "ready"
                })
            
            # Handle audio input
            elif message_type == "audio":
                if not conversation:
                    await websocket.send_json({"type": "error", "message": "Conversation not initialized"})
                    continue
                
                try:
                    # Decode audio data from hex string
                    audio_hex = data.get("audio", "")
                    if not audio_hex:
                        await websocket.send_json({"type": "error", "message": "No audio data provided"})
                        continue
                    
                    audio_bytes = bytes.fromhex(audio_hex)
                    # Skip WAV header (first 44 bytes) and convert to int16
                    audio_data = np.frombuffer(audio_bytes[44:], dtype=np.int16)
                except ValueError as e:
                    logger.error(f"Error decoding audio hex: {e}")
                    await websocket.send_json({"type": "error", "message": f"Invalid audio data: {str(e)}"})
                    continue
                
                # Add to conversation
                conversation.add_audio_input(audio_data)
                conversation.start_assistant_turn()
                
                # Generate response
                try:
                    out_text, out_audio, out_modality, text_response = generate_response(
                        conversation.chat,
                        temp=data.get("temperature", 1.0),
                        topk=data.get("top_k", 4),
                    )
                    
                    # Append to chat state
                    if out_text or out_audio:
                        conversation.chat.append(
                            text=torch.stack(out_text, 1) if out_text else torch.empty((1, 0), device=device),
                            audio_out=torch.stack(out_audio, 1) if out_audio else torch.empty((8, 0), device=device),
                            modality_flag=torch.tensor([m.value for m in out_modality], device=device).unsqueeze(0) if out_modality else torch.empty((1, 0), device=device),
                        )
                    
                    conversation.chat.end_turn()
                    conversation.chat.new_turn("user")
                    
                    # Update history
                    conversation.add_to_history("user", "[Audio message]")
                    if text_response:
                        conversation.add_to_history("assistant", text_response)
                    
                    # Send response
                    await websocket.send_json({
                        "type": "response",
                        "text": text_response,
                        "history": list(conversation.conversation_history),
                    })
                
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
            # Handle text input
            elif message_type == "text":
                if not conversation:
                    await websocket.send_json({"type": "error", "message": "Conversation not initialized"})
                    continue
                
                text_input = data.get("text", "").strip()
                if not text_input:
                    continue
                
                conversation.add_text_input(text_input)
                conversation.start_assistant_turn()
                
                try:
                    out_text, out_audio, out_modality, text_response = generate_response(
                        conversation.chat,
                        temp=data.get("temperature", 1.0),
                        topk=data.get("top_k", 4),
                    )
                    
                    if out_text or out_audio:
                        conversation.chat.append(
                            text=torch.stack(out_text, 1) if out_text else torch.empty((1, 0), device=device),
                            audio_out=torch.stack(out_audio, 1) if out_audio else torch.empty((8, 0), device=device),
                            modality_flag=torch.tensor([m.value for m in out_modality], device=device).unsqueeze(0) if out_modality else torch.empty((1, 0), device=device),
                        )
                    
                    conversation.chat.end_turn()
                    conversation.chat.new_turn("user")
                    
                    conversation.add_to_history("user", text_input)
                    if text_response:
                        conversation.add_to_history("assistant", text_response)
                    
                    await websocket.send_json({
                        "type": "response",
                        "text": text_response,
                        "history": list(conversation.conversation_history),
                    })
                
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
            # Reset conversation
            elif message_type == "reset":
                conversation = ConversationState(data.get("system_prompt", ""))
                await websocket.send_json({
                    "type": "reset_response",
                    "status": "reset"
                })
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            logger.error("Failed to send error message to client")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model": "LFM2-Audio"}


@app.get("/info")
async def model_info():
    """Get model information."""
    return {
        "model": "LFM2-Audio-1.5B",
        "capabilities": ["speech-to-speech", "text-to-speech", "interleaved"],
        "sample_rate": 24000,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8493)
