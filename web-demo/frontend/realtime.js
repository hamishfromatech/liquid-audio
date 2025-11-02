/**
 * LFM2-Audio Real-time Voice Call
 * Streaming audio with interruption support
 */

class RealtimeVoiceCall {
    constructor() {
        this.ws = null;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.audioWorklet = null;
        this.isRecording = false;
        this.isConnected = false;
        this.audioChunks = [];
        this.audioBuffer = [];
        this.isPlaying = false;
        this.hasAudioWorklet = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
        // Don't call setupAudio here - do it lazily when needed
    }
    
    initializeElements() {
        this.recordBtn = document.getElementById('recordBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.closeBtn = document.getElementById('closeBtn');
        this.resetBtn = document.getElementById('resetBtn');
        
        this.settingsModal = document.getElementById('settingsModal');
        this.closeSettingsBtn = document.getElementById('closeSettingsBtn');
        this.closeSettingsBtn2 = document.getElementById('closeSettingsBtn2');
        
        this.systemPrompt = document.getElementById('systemPrompt');
        this.temperature = document.getElementById('temperature');
        this.topK = document.getElementById('topK');
        
        this.callStatus = document.getElementById('callStatus');
        this.recordStatus = document.getElementById('recordStatus');
        this.userText = document.getElementById('userText');
        this.waveform = document.getElementById('waveform');
        this.tempValue = document.getElementById('tempValue');
        this.topKValue = document.getElementById('topKValue');
    }
    
    setupEventListeners() {
        this.recordBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
        this.settingsBtn.addEventListener('click', () => this.openSettings());
        this.closeBtn.addEventListener('click', () => this.resetConversation());
        this.resetBtn.addEventListener('click', () => this.resetConversation());
        
        this.closeSettingsBtn.addEventListener('click', () => this.closeSettings());
        this.closeSettingsBtn2.addEventListener('click', () => this.closeSettings());
        
        this.temperature.addEventListener('input', (e) => {
            this.tempValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        
        this.topK.addEventListener('input', (e) => {
            this.topKValue.textContent = e.target.value;
        });
        
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) this.closeSettings();
        });
    }
    
    async setupAudio() {
        try {
            if (this.audioContext) {
                return; // Already initialized
            }
            
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) {
                throw new Error('AudioContext not supported');
            }
            
            this.audioContext = new AudioContextClass();
            console.log('AudioContext created:', this.audioContext.state);
            
            // Resume if suspended
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
                console.log('AudioContext resumed');
            }
            
            this.hasAudioWorklet = false;
        } catch (e) {
            console.error('Error setting up audio:', e);
            throw e;
        }
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//127.0.0.1:8493/ws/call`;
        
        console.log('Connecting to:', wsUrl);
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.callStatus.textContent = 'Connected';
            this.recordStatus.textContent = 'Ready';
            this.initializeConversation();
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.callStatus.textContent = 'Connection failed';
            this.recordStatus.textContent = 'Error';
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.callStatus.textContent = 'Disconnected';
            this.recordStatus.textContent = 'Reconnecting...';
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }
    
    initializeConversation() {
        if (!this.isConnected) return;
        
        this.ws.send(JSON.stringify({
            type: 'init',
            system_prompt: this.systemPrompt.value,
            temperature: parseFloat(this.temperature.value),
            top_k: parseInt(this.topK.value)
        }));
    }
    
    async startRecording() {
        try {
            // Ensure AudioContext is initialized
            await this.setupAudio();
            
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.isRecording = true;
            
            this.recordBtn.classList.add('hidden');
            this.stopBtn.classList.remove('hidden');
            this.stopBtn.classList.add('recording');
            this.waveform.classList.remove('hidden');
            this.recordStatus.textContent = 'Listening...';
            this.userText.textContent = 'Listening...';
            
            // Use ScriptProcessor for reliable real-time audio
            this.setupScriptProcessorStream(stream);
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.recordStatus.textContent = 'Microphone access denied';
        }
    }
    
    setupScriptProcessorStream(stream) {
        try {
            const source = this.audioContext.createMediaStreamSource(stream);
            const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            processor.onaudioprocess = (event) => {
                if (this.isRecording) {
                    const inputData = event.inputBuffer.getChannelData(0);
                    this.sendAudioBuffer(Array.from(inputData));
                }
            };
            
            source.connect(processor);
            processor.connect(this.audioContext.destination);
            
            this.currentAudioProcessor = processor;
            this.currentAudioSource = source;
            
            console.log('Audio stream setup complete');
        } catch (e) {
            console.error('Error setting up audio stream:', e);
            throw e;
        }
    }
    
    stopRecording() {
        if (this.isRecording) {
            this.isRecording = false;
            
            // Stop audio streams
            if (this.currentAudioSource) {
                this.currentAudioSource.disconnect();
            }
            if (this.currentAudioProcessor) {
                this.currentAudioProcessor.disconnect();
            }
            
            this.recordBtn.classList.remove('hidden');
            this.stopBtn.classList.add('hidden');
            this.stopBtn.classList.remove('recording');
            this.waveform.classList.add('hidden');
            this.recordStatus.textContent = 'Processing...';
        }
    }
    
    sendAudioBuffer(floatData) {
        if (!this.isConnected) return;
        
        // Convert float32 to int16
        const int16Data = new Int16Array(floatData.length);
        for (let i = 0; i < floatData.length; i++) {
            int16Data[i] = Math.max(-1, Math.min(1, floatData[i])) * 0x7FFF;
        }
        
        // Convert to hex
        const hexString = Array.from(new Uint8Array(int16Data.buffer))
            .map(x => x.toString(16).padStart(2, '0'))
            .join('');
        
        this.sendAudioChunk(hexString);
    }
    
    sendAudioChunk(audioHex) {
        if (!this.isConnected) {
            this.recordStatus.textContent = 'Not connected';
            return;
        }
        
        this.recordStatus.textContent = 'Sending...';
        
        this.ws.send(JSON.stringify({
            type: 'audio_chunk',
            audio: audioHex
        }));
    }
    
    async playAudioChunk(audioHex) {
        try {
            if (!this.audioContext) {
                await this.setupAudio();
            }

            if (!audioHex || audioHex.length < 2) {
                return;
            }

            // Decode hex string to ArrayBuffer
            const byteLength = Math.floor(audioHex.length / 2);
            const buffer = new ArrayBuffer(byteLength);
            const view = new Uint8Array(buffer);
            for (let i = 0; i < byteLength; i++) {
                view[i] = parseInt(audioHex.substr(i * 2, 2), 16);
            }

            // Convert bytes to int16 PCM
            const pcm = new Int16Array(buffer);
            if (pcm.length === 0) {
                return;
            }

            // Convert int16 PCM to float32 [-1, 1]
            const float32Data = new Float32Array(pcm.length);
            for (let i = 0; i < pcm.length; i++) {
                float32Data[i] = pcm[i] / 32768.0;
            }

            // Create audio buffer and copy data
            const audioBuffer = this.audioContext.createBuffer(1, float32Data.length, 24000);
            audioBuffer.copyToChannel(float32Data, 0, 0);

            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
        } catch (e) {
            console.error('Error playing audio:', e);
        }
    }
    
    handleMessage(data) {
        const type = data.type;
        
        if (type === 'init_response') {
            this.callStatus.textContent = 'Ready';
            this.recordStatus.textContent = 'Ready';
        } else if (type === 'text_chunk') {
            // Accumulate text
            if (this.userText.textContent === 'Listening...') {
                this.userText.textContent = '';
            }
            this.userText.textContent += data.text;
        } else if (type === 'audio_chunk') {
            // Play audio chunk
            this.playAudioChunk(data.audio);
        } else if (type === 'response_complete') {
            this.recordStatus.textContent = 'Ready';
            this.userText.textContent = data.text || 'Response complete';
        } else if (type === 'error') {
            this.recordStatus.textContent = `Error: ${data.message}`;
        }
    }
    
    resetConversation() {
        this.ws.send(JSON.stringify({
            type: 'reset',
            system_prompt: this.systemPrompt.value
        }));
        
        this.userText.textContent = 'Listening...';
        this.recordStatus.textContent = 'Ready';
        this.closeSettings();
    }
    
    openSettings() {
        this.settingsModal.classList.remove('hidden');
    }
    
    closeSettings() {
        this.settingsModal.classList.add('hidden');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new RealtimeVoiceCall();
});
