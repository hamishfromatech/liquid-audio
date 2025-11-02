from collections import deque
from queue import Queue
from threading import Thread

import gradio as gr
import numpy as np
import torch
from fastrtc import AdditionalOutputs, ReplyOnPause, WebRTC

from liquid_audio import ChatState, LFMModality

from .model import device, lfm2_audio, mimi, proc


class ConversationHistory:
    """Maintains conversation history with last N messages."""
    
    def __init__(self, max_messages: int = 5):
        self.max_messages = max_messages
        self.messages: deque = deque(maxlen=max_messages)
    
    def add_user_message(self, text: str) -> None:
        """Add a user message to history."""
        self.messages.append({"role": "user", "text": text})
    
    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to history."""
        self.messages.append({"role": "assistant", "text": text})
    
    def get_formatted_history(self) -> str:
        """Get formatted conversation history for display."""
        if not self.messages:
            return "No conversation history yet."
        
        history_lines = []
        for msg in self.messages:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            role_label = "You" if msg["role"] == "user" else "Assistant"
            text = msg["text"][:100] + "..." if len(msg["text"]) > 100 else msg["text"]
            history_lines.append(f"{role_emoji} **{role_label}**: {text}")
        
        return "\n\n".join(history_lines)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


def chat_producer(
    q: Queue[torch.Tensor | None],
    chat: ChatState,
    temp: float | None,
    topk: int | None,
):
    print(f"Starting generation with state {chat}.")
    with torch.no_grad(), mimi.streaming(1):
        for t in lfm2_audio.generate_interleaved(
            **chat,
            max_new_tokens=1024,
            audio_temperature=temp,
            audio_top_k=topk,
        ):
            q.put(t)

            if t.numel() > 1:
                # Skip padding tokens (2048 is the end-of-audio token)
                if (t == 2048).any():
                    continue

                # Decode audio frame: t is shape [num_codebooks]
                # Reshape to [1, num_codebooks, 1] for mimi.decode
                wav_chunk = mimi.decode(t.unsqueeze(0).unsqueeze(-1))[0]
                q.put(wav_chunk)

    q.put(None)


def chat_response(audio: tuple[int, np.ndarray], _id: str, chat: ChatState, temp: float | None = 1.0, topk: int | None = 4, history: ConversationHistory | None = None, system_prompt: str = ""):
    if temp == 0:
        temp = None
    if topk == 0:
        topk = None

    if temp is not None:
        temp = float(temp)
    if topk is not None:
        topk = int(topk)

    # Use custom system prompt or default
    default_prompt = "Respond with interleaved text and audio. You are a helpful, friendly assistant. Keep responses concise and natural."
    final_prompt = system_prompt.strip() if system_prompt.strip() else default_prompt

    # Initialize chat on first turn
    if len(chat.text) == 1:
        chat.new_turn("system")
        chat.add_text(final_prompt)
        chat.end_turn()
        chat.new_turn("user")

    # Add user audio input
    rate, wav = audio
    chat.add_audio(torch.tensor(wav / 32_768, dtype=torch.float), rate)
    chat.end_turn()

    # Start assistant turn
    chat.new_turn("assistant")

    q: Queue[torch.Tensor | None] = Queue()
    chat_thread = Thread(target=chat_producer, args=(q, chat, temp, topk))
    chat_thread.start()

    out_text: list[torch.Tensor] = []
    out_audio: list[torch.Tensor] = []
    out_modality: list[LFMModality] = []
    audio_buffer: list[np.ndarray] = []
    buffer_threshold = 2  # Buffer at least 2 chunks before yielding
    assistant_text = ""

    while True:
        t = q.get()
        if t is None:
            break
        elif t.numel() == 1:  # text
            out_text.append(t)
            out_modality.append(LFMModality.TEXT)
            token_text = proc.text.decode(t)
            print(token_text, end="")
            assistant_text += token_text
            cur_string = proc.text.decode(torch.cat(out_text)).removesuffix("<|text_end|>")
            yield AdditionalOutputs(cur_string)
        elif t.numel() == 8:
            out_audio.append(t)
            out_modality.append(LFMModality.AUDIO_OUT)
        elif t.numel() == 1920:
            # Buffer audio chunks to smooth transitions and avoid clicks
            # t has shape [1, 1920] (batch=1, samples=1920)
            np_chunk = (t.cpu().numpy() * 32_767).astype(np.int16)
            # Squeeze batch dimension to get [1920]
            np_chunk = np_chunk.squeeze(0)
            audio_buffer.append(np_chunk)
            
            # Yield buffered audio when we have enough chunks
            if len(audio_buffer) >= buffer_threshold:
                combined = np.concatenate(audio_buffer)
                yield (24_000, combined)
                audio_buffer = []
        else:
            raise RuntimeError(f"unexpected shape: {t.shape}")
    
    # Flush remaining audio buffer
    if audio_buffer:
        combined = np.concatenate(audio_buffer)
        yield (24_000, combined)

    chat.append(
        text=torch.stack(out_text, 1),
        audio_out=torch.stack(out_audio, 1),
        modality_flag=torch.tensor([m.value for m in out_modality], device=device).unsqueeze(0),
    )

    # Update conversation history
    if history is not None:
        # Extract user message (we don't have direct access, so we'll use a placeholder)
        history.add_user_message("[Audio message]")
        # Add assistant response
        clean_text = assistant_text.removesuffix("<|text_end|>").strip()
        if clean_text:
            history.add_assistant_message(clean_text)

    chat.end_turn()
    chat.new_turn("user")


def text_chat_response(text_input: str, chat: ChatState, temp: float | None = 1.0, topk: int | None = 4, history: ConversationHistory | None = None, system_prompt: str = ""):
    """Handle text-based chat input."""
    if temp == 0:
        temp = None
    if topk == 0:
        topk = None

    if temp is not None:
        temp = float(temp)
    if topk is not None:
        topk = int(topk)

    if not text_input.strip():
        return "", ""

    # Use custom system prompt or default
    default_prompt = "Respond with interleaved text and audio. You are a helpful, friendly assistant. Keep responses concise and natural."
    final_prompt = system_prompt.strip() if system_prompt.strip() else default_prompt

    # Initialize chat on first turn
    if len(chat.text) == 1:
        chat.new_turn("system")
        chat.add_text(final_prompt)
        chat.end_turn()
        chat.new_turn("user")

    # Add user text input
    chat.add_text(text_input)
    chat.end_turn()

    # Start assistant turn
    chat.new_turn("assistant")

    q: Queue[torch.Tensor | None] = Queue()
    chat_thread = Thread(target=chat_producer, args=(q, chat, temp, topk))
    chat_thread.start()

    out_text: list[torch.Tensor] = []
    out_audio: list[torch.Tensor] = []
    out_modality: list[LFMModality] = []
    assistant_text = ""

    while True:
        t = q.get()
        if t is None:
            break
        elif t.numel() == 1:  # text
            out_text.append(t)
            out_modality.append(LFMModality.TEXT)
            token_text = proc.text.decode(t)
            print(token_text, end="")
            assistant_text += token_text
        elif t.numel() == 8:
            out_audio.append(t)
            out_modality.append(LFMModality.AUDIO_OUT)
        elif t.numel() == 1920:
            pass  # Skip audio chunks for text chat
        else:
            raise RuntimeError(f"unexpected shape: {t.shape}")

    chat.append(
        text=torch.stack(out_text, 1),
        audio_out=torch.stack(out_audio, 1) if out_audio else torch.empty((8, 0), device=device),
        modality_flag=torch.tensor([m.value for m in out_modality], device=device).unsqueeze(0),
    )

    # Update conversation history
    if history is not None:
        history.add_user_message(text_input)
        clean_text = assistant_text.removesuffix("<|text_end|>").strip()
        if clean_text:
            history.add_assistant_message(clean_text)

    chat.end_turn()
    chat.new_turn("user")

    return assistant_text.removesuffix("<|text_end|>").strip(), history.get_formatted_history() if history else ""


def clear(history: ConversationHistory):
    gr.Info("Cleared chat history", duration=3)
    history.clear()
    return ChatState(proc), None, history.get_formatted_history()


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css="""
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .header-container h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .header-container p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 0.95rem;
    }
    .control-panel {
        background: rgba(102, 126, 234, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .output-section {
        background: rgba(0, 0, 0, 0.02);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 0, 0, 0.08);
    }
    .button-group {
        display: flex;
        gap: 0.75rem;
        margin-top: 1rem;
    }
    .history-container {
        background: rgba(102, 126, 234, 0.08);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        max-height: 300px;
        overflow-y: auto;
    }
    """
) as demo, gr.Column():
        # Header
        with gr.Group(elem_classes="header-container"):
            gr.Markdown("# 🎙️ LFM2-Audio Speech Chat")
            gr.Markdown("*Real-time speech-to-speech conversation with interleaved text and audio responses*")

        chat_state = gr.State(ChatState(proc))
        conversation_history = gr.State(ConversationHistory(max_messages=5))
        
        # Control Panel
        with gr.Group(elem_classes="control-panel"):
            gr.Markdown("### Settings")
            
            # System Prompt
            system_prompt = gr.Textbox(
                value="Respond with interleaved text and audio. You are a helpful, friendly assistant. Keep responses concise and natural.",
                label="System Prompt",
                info="Instructions for the AI. Leave empty to use default.",
                lines=3,
                placeholder="Enter custom instructions for the AI...",
            )
            
            with gr.Row():
                temp_slider = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more deterministic",
                )
                topk_slider = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=4,
                    step=1,
                    label="Top-K",
                    info="Number of top candidates to sample from",
                )
        
        # Conversation History Display
        gr.Markdown("### 📜 Conversation History (Last 5 messages)")
        with gr.Group(elem_classes="history-container"):
            history_display = gr.Markdown(
                "No conversation history yet.",
                label="History",
            )
        
        # Input Tabs
        with gr.Tabs():
            # Audio Input Tab
            with gr.TabItem("🎤 Voice Chat"):
                gr.Markdown("Speak to the model and get audio responses")
                webrtc = WebRTC(
                    modality="audio",
                    mode="send-receive",
                    full_screen=False,
                )
            
            # Text Input Tab
            with gr.TabItem("💬 Text Chat"):
                gr.Markdown("Chat with the model using text")
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here...",
                        lines=2,
                    )
                    text_submit = gr.Button("Send", variant="primary", size="lg")
        
        # Output Section
        gr.Markdown("### 📝 Current Response")
        with gr.Group(elem_classes="output-section"):
            text_out = gr.Textbox(
                lines=5,
                label="Text Output",
                interactive=False,
                show_label=True,
            )
        
        # Controls
        with gr.Row():
            clear_btn = gr.Button("🔄 Reset Chat", variant="secondary", size="lg")
            info_btn = gr.Button("i About", variant="secondary", size="lg")

        def update_history_display(history: ConversationHistory):
            return history.get_formatted_history()

        webrtc.stream(
            ReplyOnPause(
                chat_response,  # type: ignore[arg-type]
                input_sample_rate=24_000,
                output_sample_rate=24_000,
                can_interrupt=False,
            ),
            inputs=[webrtc, chat_state, temp_slider, topk_slider, conversation_history, system_prompt],
            outputs=[webrtc],
        )
        webrtc.on_additional_outputs(
            lambda s: s,
            outputs=[text_out],
        )
        
        # Text chat submit button
        text_submit.click(
            text_chat_response,
            inputs=[text_input, chat_state, temp_slider, topk_slider, conversation_history, system_prompt],
            outputs=[text_out, history_display],
        ).then(
            lambda: "",
            outputs=[text_input],
        )
        
        # Update history display after each response
        def on_response_complete(history: ConversationHistory):
            return history.get_formatted_history()
        
        clear_btn.click(
            clear,
            inputs=[conversation_history],
            outputs=[chat_state, text_out, history_display]
        )
        
        # About modal
        info_btn.click(
            lambda: gr.Info(
                "LFM2-Audio is a speech-to-speech model that can generate interleaved text and audio responses. "
                "Pause after speaking to get a response. Adjust temperature and top-k to control output diversity. "
                "The conversation history shows your last 5 messages for context.",
                duration=5
            )
        )


def main():
    demo.launch()


if __name__ == "__main__":
    main()
