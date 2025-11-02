/**
 * LFM2-Audio Web Voice Call Interface
 * Minimal, clean design with focus on voice interaction
 */

class VoiceCallApp {
    constructor() {
        this.ws = null;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.isRecording = false;
        this.isConnected = false;
        this.audioChunks = [];
        this.messages = [];
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
    }
    
    initializeElements() {
        // Buttons
        this.recordBtn = document.getElementById('recordBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.closeBtn = document.getElementById('closeBtn');
        this.resetBtn = document.getElementById('resetBtn');
        
        // Modals
        this.settingsModal = document.getElementById('settingsModal');
        this.chatModal = document.getElementById('chatModal');
        this.closeSettingsBtn = document.getElementById('closeSettingsBtn');
        this.closeSettingsBtn2 = document.getElementById('closeSettingsBtn2');
        this.closeChatBtn = document.getElementById('closeChatBtn');
        
        // Inputs
        this.systemPrompt = document.getElementById('systemPrompt');
        this.temperature = document.getElementById('temperature');
        this.topK = document.getElementById('topK');
        
        // Display
        this.chatContainer = document.getElementById('chatContainer');
        this.recordStatus = document.getElementById('recordStatus');
        this.tempValue = document.getElementById('tempValue');
        this.topKValue = document.getElementById('topKValue');
    }
    
    setupEventListeners() {
        this.recordBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
        this.settingsBtn.addEventListener('click', () => this.openSettings());
        this.closeBtn.addEventListener('click', () => this.openChat());
        this.resetBtn.addEventListener('click', () => this.resetConversation());
        
        this.closeSettingsBtn.addEventListener('click', () => this.closeSettings());
        this.closeSettingsBtn2.addEventListener('click', () => this.closeSettings());
        this.closeChatBtn.addEventListener('click', () => this.closeChat());
        
        this.temperature.addEventListener('input', (e) => {
            this.tempValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        
        this.topK.addEventListener('input', (e) => {
            this.topKValue.textContent = e.target.value;
        });
        
        // Close modals on background click
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) this.closeSettings();
        });
        
        this.chatModal.addEventListener('click', (e) => {
            if (e.target === this.chatModal) this.closeChat();
        });
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Connect to backend on port 8493
        const wsUrl = `${protocol}//127.0.0.1:8493/ws/chat`;
        
        console.log('Attempting to connect to:', wsUrl);
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
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
            this.recordStatus.textContent = 'Connection failed - check console';
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.recordStatus.textContent = 'Disconnected - retrying...';
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }
    
    initializeConversation() {
        if (!this.isConnected) return;
        
        this.ws.send(JSON.stringify({
            type: 'init',
            system_prompt: this.systemPrompt.value
        }));
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                this.processAudio();
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            this.recordBtn.classList.add('hidden');
            this.stopBtn.classList.remove('hidden');
            this.stopBtn.classList.add('recording');
            this.recordStatus.textContent = 'Listening...';
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.recordStatus.textContent = 'Microphone access denied';
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            this.recordBtn.classList.remove('hidden');
            this.stopBtn.classList.add('hidden');
            this.stopBtn.classList.remove('recording');
            this.recordStatus.textContent = 'Processing...';
        }
    }
    
    processAudio() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        const reader = new FileReader();
        
        reader.onload = () => {
            const arrayBuffer = reader.result;
            // Convert to Uint8Array to get raw bytes
            const uint8Array = new Uint8Array(arrayBuffer);
            // Convert each byte to hex string
            const hexString = Array.from(uint8Array)
                .map(x => x.toString(16).padStart(2, '0'))
                .join('');
            
            this.sendAudio(hexString);
        };
        
        reader.readAsArrayBuffer(audioBlob);
    }
    
    sendAudio(audioHex) {
        if (!this.isConnected) {
            this.recordStatus.textContent = 'Not connected';
            return;
        }
        
        this.recordStatus.textContent = 'Sending...';
        
        this.ws.send(JSON.stringify({
            type: 'audio',
            audio: audioHex,
            temperature: parseFloat(this.temperature.value),
            top_k: parseInt(this.topK.value)
        }));
    }
    
    resetConversation() {
        this.messages = [];
        this.updateChatDisplay();
        
        this.ws.send(JSON.stringify({
            type: 'reset',
            system_prompt: this.systemPrompt.value
        }));
        
        this.closeSettings();
    }
    
    handleMessage(data) {
        const type = data.type;
        
        if (type === 'init_response' || type === 'reset_response') {
            this.recordStatus.textContent = 'Ready';
        } else if (type === 'response') {
            this.messages.push({ role: 'user', text: '[Audio message]' });
            this.messages.push({ role: 'assistant', text: data.text });
            this.updateChatDisplay();
            this.recordStatus.textContent = 'Ready';
        } else if (type === 'error') {
            this.recordStatus.textContent = `Error: ${data.message}`;
        }
    }
    
    updateChatDisplay() {
        if (this.messages.length === 0) {
            this.chatContainer.innerHTML = '<p class="text-center text-gray-500 py-8">No messages yet</p>';
            return;
        }
        
        this.chatContainer.innerHTML = this.messages.map(msg => `
            <div class="space-y-2">
                <p class="text-sm font-semibold text-gray-600">${msg.role === 'user' ? 'You' : 'Assistant'}</p>
                <p class="text-sm text-gray-700 bg-gray-100 p-3 rounded-lg">${this.escapeHtml(msg.text)}</p>
            </div>
        `).join('');
        
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    openSettings() {
        this.settingsModal.classList.remove('hidden');
    }
    
    closeSettings() {
        this.settingsModal.classList.add('hidden');
    }
    
    openChat() {
        this.chatModal.classList.remove('hidden');
    }
    
    closeChat() {
        this.chatModal.classList.add('hidden');
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new VoiceCallApp();
});
