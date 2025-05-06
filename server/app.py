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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # <-- allow all origins temporarily


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
        audio_file.save("uploaded_audio.wav")
        
        user_input = transcribe_audio("uploaded_audio.wav")
        if not user_input.strip():
            return jsonify({"error": "Empty transcription"}), 400

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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
