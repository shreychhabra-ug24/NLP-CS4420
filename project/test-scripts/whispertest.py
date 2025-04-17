import whisper
import os
import ssl
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Function to record live audio
def record_audio(output_file, duration=5, sample_rate=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write("temp.wav", sample_rate, audio_data)  # Save as WAV file
    print("Recording finished. Saving as MP3...")
    
    # Convert WAV to MP3
    audio = AudioSegment.from_wav("temp.wav")
    audio.export(output_file, format="mp3")
    os.remove("temp.wav")  # Clean up temporary WAV file
    print(f"Audio saved as {output_file}")

# Record live audio and save it as an MP3
output_mp3 = "live_recording.mp3"
record_audio(output_mp3, duration=5)  # Record for 5 seconds

# Load the Whisper model
model = whisper.load_model("turbo")

# Transcribe the recorded audio
result = model.transcribe(output_mp3)
print("Transcription:")
print(result["text"])