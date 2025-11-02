# LFM2-Audio Web Demo

A modern, production-ready web interface for LFM2-Audio voice conversations. Features a beautiful UI built with Tailwind CSS and a robust FastAPI backend for real-time speech-to-speech interactions.

## Features

- 🎤 **Real-time Voice Chat**: Speak to the AI and get instant audio responses
- 💬 **Text Chat**: Type messages for faster interactions
- 🎨 **Modern UI**: Beautiful gradient design with glass-morphism effects
- ⚙️ **Customizable**: Adjust temperature, top-k, and system prompts
- 📱 **Responsive**: Works on desktop and mobile devices
- 🔌 **API-First**: Separate backend API that can be integrated into any web app
- 🚀 **Production Ready**: Built with FastAPI and WebSockets for real-time communication

## Architecture

```
web-demo/
├── backend/
│   ├── api.py              # FastAPI WebSocket server
│   └── requirements.txt    # Python dependencies
└── frontend/
    ├── index.html          # Modern UI with Tailwind CSS
    ├── app.js              # Real-time WebSocket client
    └── README.md           # This file
```

## Quick Start

### Backend Setup

1. **Install dependencies**:
```bash
cd web-demo/backend
pip install -r requirements.txt
```

2. **Run the API server**:
```bash
python api.py
```

The server will start on `http://localhost:8000`

### Frontend Setup

1. **Serve the frontend** (use any HTTP server):
```bash
cd web-demo/frontend

# Using Python 3
python -m http.server 8080

# Or using Node.js
npx http-server -p 8080
```

2. **Open in browser**:
Navigate to `http://localhost:8080`

## API Endpoints

### WebSocket: `/ws/chat`

Real-time bidirectional communication for voice and text chat.

**Message Types:**

#### Initialize Conversation
```json
{
  "type": "init",
  "system_prompt": "You are a helpful assistant..."
}
```

#### Send Audio
```json
{
  "type": "audio",
  "audio": "hex_encoded_audio_data",
  "temperature": 1.0,
  "top_k": 4
}
```

#### Send Text
```json
{
  "type": "text",
  "text": "Your message here",
  "temperature": 1.0,
  "top_k": 4
}
```

#### Reset Conversation
```json
{
  "type": "reset",
  "system_prompt": "New system prompt..."
}
```

**Response Format:**
```json
{
  "type": "response",
  "text": "Assistant's response",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### REST: `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "LFM2-Audio"
}
```

### REST: `/info`

Get model information.

**Response:**
```json
{
  "model": "LFM2-Audio-1.5B",
  "capabilities": ["speech-to-speech", "text-to-speech", "interleaved"],
  "sample_rate": 24000
}
```

## Configuration

### Backend Configuration

Edit `api.py` to customize:
- **Host/Port**: Change in `uvicorn.run()` call
- **CORS**: Modify `CORSMiddleware` settings
- **Model paths**: Update import paths if needed

### Frontend Configuration

Edit `app.js` to customize:
- **WebSocket URL**: Modify `connectWebSocket()` method
- **UI colors**: Edit Tailwind classes in `index.html`
- **Default settings**: Change default values in `initializeElements()`

## Integration with Other Web Apps

The API is designed to be framework-agnostic. You can integrate it with any web framework:

### React Example
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  setResponse(data.text);
};

const sendMessage = (text) => {
  ws.send(JSON.stringify({
    type: 'text',
    text: text
  }));
};
```

### Vue Example
```javascript
export default {
  data() {
    return { ws: null };
  },
  mounted() {
    this.ws = new WebSocket('ws://localhost:8000/ws/chat');
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.response = data.text;
    };
  }
}
```

## Performance Tips

1. **Audio Quality**: Use 16-bit PCM at 24kHz for best results
2. **Latency**: Keep WebSocket connection alive for minimal latency
3. **Buffering**: The frontend buffers audio chunks for smooth playback
4. **Temperature**: Lower values (0.5-1.0) for more consistent responses
5. **Top-K**: Higher values (8-12) for more creative responses

## Troubleshooting

### WebSocket Connection Failed
- Ensure backend is running on the correct port
- Check CORS settings if frontend is on different domain
- Verify firewall allows WebSocket connections

### Audio Not Recording
- Check browser microphone permissions
- Ensure HTTPS is used (required for getUserMedia in production)
- Test microphone in browser settings

### Slow Responses
- Check GPU memory usage
- Reduce `max_new_tokens` in backend if needed
- Consider using CPU device for M2 Max (see memory note)

## Deployment

### Docker (Recommended)

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/api.py .
COPY frontend /app/frontend

EXPOSE 8000

CMD ["python", "api.py"]
```

Build and run:
```bash
docker build -t lfm2-audio-web .
docker run -p 8000:8000 lfm2-audio-web
```

### Production Considerations

1. **HTTPS**: Use a reverse proxy (nginx) with SSL
2. **Authentication**: Add API key validation
3. **Rate Limiting**: Implement request throttling
4. **Monitoring**: Add logging and metrics
5. **Scaling**: Use load balancer for multiple backend instances

## License

Same as liquid-audio package

## Support

For issues and feature requests, visit the main liquid-audio repository.
