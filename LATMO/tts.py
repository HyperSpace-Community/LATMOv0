from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import os

def initialize_pipeline():
    """Initialize TTS pipeline with American English"""
    return KPipeline(lang_code='a')

def get_unique_filename(base_filename):
    """Generate unique filename by adding increment number if file exists."""
    if not os.path.exists(base_filename):
        return base_filename
    
    name, ext = os.path.splitext(base_filename)
    counter = 1
    while os.path.exists(f"{name}_{counter}{ext}"):
        counter += 1
    return f"{name}_{counter}{ext}"

def speak_text(text, voice='am_puck', speed=1.2, output_dir=None):
    """
    Generate speech from text and save audio segments.
    
    Args:
        text (str): Text to convert to speech
        voice (str): Voice to use (default: 'am_puck')
        speed (float): Speech speed (default: 1.2)
        output_dir (str): Directory to save audio files (default: None)
    
    Returns:
        list: List of paths to generated audio segments
    """
    try:
        pipeline = initialize_pipeline()
        
        # Set default output directory if none provided
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'output')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate speech segments
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
        
        segment_files = []
        for i, (gs, ps, audio) in enumerate(generator):
            # Save individual segment
            base_segment_file = os.path.join(output_dir, f"segment_{i}.wav")
            segment_file = get_unique_filename(base_segment_file)
            
            try:
                sf.write(segment_file, audio, 24000)
                segment_files.append(segment_file)
                print(f"Saved segment {i} to: {segment_file}")
            except Exception as save_error:
                print(f"Error saving segment {i}: {str(save_error)}")
                continue
        
        return segment_files
    
    except Exception as e:
        print(f"Error in speech generation: {str(e)}")
        return []

# Example usage when run directly
if __name__ == "__main__":
    demo_text = "Hello, this is a test of the text-to-speech system."
    segments = speak_text(demo_text)
    print(f"Generated {len(segments)} audio segments")