from flask import Flask, request, jsonify
from flask_cors import CORS
import sounddevice as sd
import numpy as np
import soundfile as sf
import torch
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS
import librosa
import logging
import os

import subprocess

def convert_to_wav(input_path, output_path):
    # Use ffmpeg to convert any audio to PCM WAV
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ], check=True)

# Initialize Flask app
app = Flask(__name__)

# ALLOW React frontend access
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)

# Global variables
history = []
sample_rate = 16000
max_turns = 5
speaker_wav = "samplevoice.wav"
language = "en"

def transcribe_audio(file_path):
    try:
        audio_input, sr = sf.read(file_path)
        if sr != sample_rate:
            audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=sample_rate)
        if audio_input.ndim > 1:
            audio_input = np.mean(audio_input, axis=1)
        input_features = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

@app.route("/chat", methods=["POST"])
def chat():
    global history
    
    try:
        logger.info("Request received with files: %s", request.files.keys())
        
        if "audio" not in request.files:
            logger.error("No audio file in request")
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files["audio"]
        
        # Save original file with temporary name
        temp_path = "uploaded_audio_temp"
        audio_file.save(temp_path)
        
        try:
            # Convert to proper WAV format
            convert_to_wav(temp_path, "uploaded_audio.wav")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {str(e)}")
            return jsonify({"error": "Audio format conversion failed"}), 400
            
        # Now process the converted WAV file
        user_input = transcribe_audio("uploaded_audio.wav")
        
        # Cleanup temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not user_input.strip():
            return jsonify({"error": "Empty transcription"}), 400

        # Rest of your existing code remains the same...
        history.append(f"User: {user_input}")
        history = history[-max_turns * 2:]  # Trim history
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": "\n".join(history) + "\nAI:", "stream": False}
        )
        response.raise_for_status()
        ai_response = response.json().get("response", "[No response]")
        
        history.append(f"AI: {ai_response}")

        # Generate speech using gTTS
        tts = gTTS(text=ai_response, lang='en')
        tts.save("reply.mp3")

        return jsonify({"reply": ai_response, "history": history})

    except Exception as e:
        logger.exception("Error in chat endpoint")
        return jsonify({"error": str(e)}), 500

# Change Flask port instead of fighting SIP
if __name__ == "__main__":
    app.run(port=8000)  # In app.py
