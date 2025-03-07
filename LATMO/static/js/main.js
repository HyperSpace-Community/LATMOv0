document.addEventListener('DOMContentLoaded', () => {
    // Initialize Socket.io
    const socket = io();
    
    // DOM Elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearButton = document.getElementById('clear-button');
    const ttsToggle = document.getElementById('tts-toggle');
    const voiceSelect = document.getElementById('voice-select');
    const speedRange = document.getElementById('speed-range');
    const speedValue = document.getElementById('speed-value');
    
    // Audio player for TTS
    let currentAudio = null;
    
    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    });
    
    // Send message function
    const sendMessage = () => {
        const message = userInput.value.trim();
        if (message) {
            // Add user message to chat
            addMessage('user', message);
            
            // Send message to server
            socket.emit('send_message', { message });
            
            // Clear input
            userInput.value = '';
            userInput.style.height = 'auto';
        }
    };
    
    // Handle send button click
    sendButton.addEventListener('click', sendMessage);
    
    // Handle enter key (with shift+enter for new line)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Clear chat
    clearButton.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear the chat?')) {
            chatMessages.innerHTML = '';
            if (currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
        }
    });
    
    // Handle received messages
    socket.on('receive_message', (data) => {
        addMessage('assistant', data.response);
        
        // Handle TTS if enabled
        if (ttsToggle.checked && data.audio_files && data.audio_files.length > 0) {
            playAudioSequence(data.audio_files);
        }
    });
    
    // Add message to chat
    const addMessage = (type, content) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<i class="fas fa-robot"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };
    
    // Play audio sequence
    const playAudioSequence = (audioFiles) => {
        let currentIndex = 0;
        
        const playNext = () => {
            if (currentIndex < audioFiles.length) {
                if (currentAudio) {
                    currentAudio.pause();
                }
                
                currentAudio = new Audio(`/audio/${audioFiles[currentIndex]}`);
                currentAudio.onended = () => {
                    currentIndex++;
                    playNext();
                };
                currentAudio.play().catch(console.error);
            }
        };
        
        playNext();
    };
    
    // Handle TTS toggle
    ttsToggle.addEventListener('change', () => {
        fetch('/toggle_tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!ttsToggle.checked && currentAudio) {
            currentAudio.pause();
            currentAudio = null;
        }
    });
    
    // Handle voice and speed changes
    const updateTTSSettings = () => {
        fetch('/update_tts_settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                voice: voiceSelect.value,
                speed: speedRange.value
            })
        });
    };
    
    voiceSelect.addEventListener('change', updateTTSSettings);
    speedRange.addEventListener('input', () => {
        speedValue.textContent = `${speedRange.value}x`;
        updateTTSSettings();
    });
    
    // Add loading animation
    const showLoading = () => {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant loading';
        loadingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return loadingDiv;
    };
    
    // Socket connection handling
    socket.on('connect', () => {
        console.log('Connected to server');
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        addMessage('system', 'Disconnected from server. Trying to reconnect...');
    });
    
    // Initialize with a welcome message
    addMessage('assistant', `Hello! I'm ${window.assistantName}. How can I help you today?`);
}); 