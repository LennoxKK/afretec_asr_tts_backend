from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch, io
import soundfile as sf
from pydub import AudioSegment
from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoModelForTextToWaveform
import os
import numpy as np
import subprocess
import sys

app = Flask(__name__)
CORS(app)

device = "cpu"

# ===== CHECK AND CREATE QUANTIZED MODEL ===== #
def ensure_quantized_model_exists():
    """Run the quantization script if quantized model doesn't exist"""
    asr_model_path = "./quantized_asr_model"
    
    if not os.path.exists(asr_model_path):
        print("📦 Quantized ASR model not found. Running quantization script...")
        try:
            # Run the quantization script as a separate process
            result = subprocess.run([sys.executable, "fix_quantization.py"], 
                                  capture_output=True, text=True, check=True)
            print("✅ Quantization script completed successfully")
            print("Output:", result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"❌ Quantization script failed: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise Exception("Failed to create quantized model")
    else:
        print("✅ Quantized ASR model already exists")

# ===== LOAD MODELS ===== #
# Ensure quantized model exists before loading
ensure_quantized_model_exists()

print("🔄 Loading quantized ASR model...")
try:
    processor = AutoProcessor.from_pretrained("./quantized_asr_model")
    # Don't use .to() for 8-bit models - they're already properly configured
    asr_model = AutoModelForCTC.from_pretrained(
        "./quantized_asr_model", 
        torch_dtype=torch.float32
    ).eval()  # Remove .to(device) for 8-bit models
    print("✅ Quantized ASR model loaded successfully")
except Exception as e:
    print(f"❌ Error loading quantized ASR model: {e}")
    raise

print("🔄 Loading TTS models...")
fallback_tokenizer = AutoTokenizer.from_pretrained("Humphery7/tts-models-yoruba")
fallback_model = AutoModelForTextToWaveform.from_pretrained("Humphery7/tts-models-yoruba").to(device).eval()
tts_tokenizer = AutoTokenizer.from_pretrained("Workhelio/yoruba_tts")
tts_model = AutoModelForTextToWaveform.from_pretrained("Workhelio/yoruba_tts").to(device).eval()
print("✅ TTS models loaded")

# ===== HELPERS ===== #
def convert_to_wav_bytes(file_bytes: bytes) -> bytes:
    """Convert uploaded audio to WAV bytes (mono 16kHz)."""
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)
    return buf.read()

def preprocess_audio(file_bytes: bytes):
    """Load WAV bytes into torch tensor for ASR."""
    wav_bytes = convert_to_wav_bytes(file_bytes)
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return torch.tensor(data, dtype=torch.float32), sr

def transcribe_audio_chunks(audio_data, sr, chunk_length_s=30):
    """
    Transcribe audio by splitting into chunks to handle long audio files.
    """
    # Convert to numpy array if it's a tensor
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.numpy()
    
    duration = len(audio_data) / sr
    print(f"   📊 Audio duration: {duration:.2f}s | Sample rate: {sr} Hz")
    
    # Handle long audio via chunking
    max_samples = int(chunk_length_s * sr)
    chunks = [audio_data[i:i + max_samples] for i in range(0, len(audio_data), max_samples)]
    
    if len(chunks) > 1:
        print(f"   🔹 Splitting into {len(chunks)} chunk(s) of {chunk_length_s}s each")
    
    all_text = []
    
    with torch.no_grad():
        for idx, chunk in enumerate(chunks, start=1):
            try:
                # Preprocess directly to tensor - let the processor handle device placement
                inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
                
                # For 8-bit models, don't move inputs to device - let the model handle it
                # inputs = {k: v.to(device, dtype=torch.float32) for k, v in inputs.items()}

                # Forward pass
                logits = asr_model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=-1)
                text = processor.batch_decode(pred_ids)[0].strip()
                all_text.append(text)
                
                if len(chunks) > 1:
                    print(f"      🧩 Chunk {idx}/{len(chunks)} processed")
            except Exception as e:
                print(f"      ❌ Error processing chunk {idx}: {e}")
                all_text.append("")  # Add empty string for failed chunk
    
    # Merge final transcription
    transcription = " ".join(all_text).replace("  ", " ").strip()
    return transcription

def normalize_audio(waveform):
    """Normalize audio waveform to prevent clipping."""
    waveform_np = waveform.squeeze().cpu().numpy()
    max_val = np.max(np.abs(waveform_np))
    if max_val > 0:
        waveform_np = waveform_np / max_val * 0.9
    return waveform_np

# ===== ENDPOINTS ===== #
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        print(f"🎵 Processing audio file: {file.filename}")
        speech, sr = preprocess_audio(file.read())
        text = transcribe_audio_chunks(speech, sr)
        return jsonify({"transcription": text})
    except Exception as e:
        print("Transcription error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing text"}), 400

    text = data["text"]
    print(f"🎵 TTS Request: '{text}'")
    
    try:
        # Try main model first
        inputs = tts_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = tts_model(**inputs)
        
        waveform = normalize_audio(out.waveform)
        sr = tts_model.config.sampling_rate
        print("✅ TTS generated with main model (Humphery7)")
        
    except Exception as e:
        print(f"⚠ Primary TTS failed, falling back: {e}")
        try:
            # Try fallback model
            inputs = fallback_tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = fallback_model(**inputs)
            
            waveform = normalize_audio(out.waveform)
            sr = fallback_model.config.sampling_rate
            print("✅ TTS generated with fallback model (Workhelio)")
            
        except Exception as e2:
            print(f"❌ Fallback TTS also failed: {e2}")
            return jsonify({"error": f"TTS failed: {e2}"}), 500

    # ===== Stream waveform in-memory ===== #
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav", as_attachment=True, download_name="tts_output.wav")

# ===== Health & root ===== #
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Backend is running"})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Yoruba ASR/TTS Backend", "status": "running"})

# ===== Model Management Endpoints ===== #
@app.route("/models/recreate-asr", methods=["POST"])
def recreate_asr_model():
    """Force recreate the ASR model by running quantization script again"""
    try:
        print("🔄 Manually recreating ASR model...")
        result = subprocess.run([sys.executable, "fix_quantization.py"], 
                              capture_output=True, text=True, check=True)
        
        # Reload the model
        global processor, asr_model
        processor = AutoProcessor.from_pretrained("./quantized_asr_model")
        asr_model = AutoModelForCTC.from_pretrained("./quantized_asr_model", torch_dtype=torch.float32).eval()
        
        return jsonify({
            "status": "success", 
            "message": "ASR model recreated successfully",
            "output": result.stdout
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": f"Failed to recreate ASR model: {e}",
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500

@app.route("/models/info", methods=["GET"])
def model_info():
    """Return information about loaded models"""
    asr_model_exists = os.path.exists("./quantized_asr_model")
    info = {
        "asr_model": "quantized_asr_model (local)" if asr_model_exists else "not found",
        "tts_models": {
            "primary": "Humphery7/tts-models-yoruba (original)",
            "fallback": "Workhelio/yoruba_tts (original)"
        },
        "device": device,
        "asr_model_exists": asr_model_exists
    }
    return jsonify(info)

# ===== RUN ===== #
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("🚀 Yoruba ASR/TTS Backend Starting...")
    print(f"{'='*60}")
    print(f"📁 ASR Model: quantized (created by fix_quantization.py)")
    print(f"🔊 TTS Models: Original HuggingFace models")
    print(f"⚡ Device: {device}")
    print(f"{'='*60}\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)