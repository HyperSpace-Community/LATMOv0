from flask import Flask, render_template, request, jsonify, send_file
import os
from main import process_message, SYSTEM_CONFIG, speak_text, SYSTEM_INSTRUCTIONS
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Add route to serve audio files
@app.route('/audio/<path:filename>')
def serve_audio(filename):
    audio_path = os.path.join(os.path.dirname(__file__), 'output', filename)
    return send_file(audio_path, mimetype='audio/wav')

@app.route('/')
def home():
    # Pass all system configuration data to the template
    return render_template('index.html', 
                         assistant_name=SYSTEM_CONFIG['assistant_name'],
                         assistant_definition=SYSTEM_CONFIG['assistant_definition'],
                         created_by=SYSTEM_CONFIG['created_by'],
                         purpose=SYSTEM_CONFIG['purpose'],
                         mission=SYSTEM_CONFIG['mission'],
                         core_values=SYSTEM_CONFIG['core_values'],
                         capabilities=SYSTEM_CONFIG['capabilities'],
                         priorities=SYSTEM_CONFIG['priorities'],
                         automation_focus=SYSTEM_CONFIG['automation_focus'],
                         interaction_guidelines=SYSTEM_CONFIG['interaction_guidelines'])

@socketio.on('connect')
def handle_connect():
    # Send initial greeting
    intro_text = f"Greetings! I am {SYSTEM_CONFIG['assistant_name']}. {SYSTEM_CONFIG['mission']}"
    emit('receive_message', {
        'response': f"{SYSTEM_CONFIG['assistant_name']}: {intro_text}",
        'assistant_name': SYSTEM_CONFIG['assistant_name'],
        'audio_files': []
    })

@socketio.on('send_message')
def handle_message(data):
    user_message = data['message']
    
    # Prepend system instructions to ensure proper identity
    full_message = f"{SYSTEM_INSTRUCTIONS}\n\nUser message: {user_message}"
    
    # Process the message
    response = process_message(full_message)
    
    # Ensure response starts with LATMO's name
    if not response.startswith(f"{SYSTEM_CONFIG['assistant_name']}:"):
        response = f"{SYSTEM_CONFIG['assistant_name']}: {response}"
    
    # Replace any AI model references
    response = response.replace("I am an AI language model", f"I am {SYSTEM_CONFIG['assistant_name']}")
    response = response.replace("I am an artificial intelligence", f"I am {SYSTEM_CONFIG['assistant_name']}")
    response = response.replace("I am a language model", f"I am {SYSTEM_CONFIG['assistant_name']}")
    response = response.replace("I am an AI assistant", f"I am {SYSTEM_CONFIG['assistant_name']}")
    
    # Generate speech if TTS is enabled
    audio_files = []
    if SYSTEM_CONFIG.get('tts_enabled', True):
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        # Clear previous audio files
        for file in os.listdir(output_dir):
            if file.startswith('response_') and file.endswith('.wav'):
                try:
                    os.remove(os.path.join(output_dir, file))
                except:
                    pass
        
        # Generate new speech
        speak_text(
            response,
            voice=SYSTEM_CONFIG.get('tts_voice', 'am_puck'),
            speed=SYSTEM_CONFIG.get('tts_speed', 1.2)
        )
        
        # Get list of new audio files
        audio_files = [f for f in os.listdir(output_dir) 
                      if f.startswith('response_') and f.endswith('.wav')]
        audio_files.sort()  # Ensure files are in order
    
    # Emit the response back to the client
    emit('receive_message', {
        'response': response,
        'assistant_name': SYSTEM_CONFIG['assistant_name'],
        'audio_files': audio_files
    })

@app.route('/toggle_tts', methods=['POST'])
def toggle_tts():
    SYSTEM_CONFIG['tts_enabled'] = not SYSTEM_CONFIG.get('tts_enabled', True)
    return jsonify({'tts_enabled': SYSTEM_CONFIG['tts_enabled']})

@app.route('/update_tts_settings', methods=['POST'])
def update_tts_settings():
    data = request.json
    SYSTEM_CONFIG['tts_voice'] = data.get('voice', 'am_puck')
    SYSTEM_CONFIG['tts_speed'] = float(data.get('speed', 1.2))
    return jsonify({'success': True})

if __name__ == '__main__':
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'), exist_ok=True)
    socketio.run(app, debug=True) 