import os
import torch
import librosa
import numpy as np
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification
from spaces import GPU
import zipfile

base_dir = os.path.dirname(__file__)
zip_path = os.path.join(base_dir, "models", "whisper-deepfake-detection-8bit.zip")
extracted_model_path = os.path.join(base_dir, "models", "whisper-deepfake-detection-8bit-extracted")
fallback_model_path = os.path.join(base_dir, "models", "whisper-deepfake-detection-8bit")
nested_path = os.path.join(extracted_model_path, "whisper-deepfake-detection-8bit")

if os.path.exists(zip_path) and not os.path.exists(extracted_model_path):
    print("[DEBUG] Extracting model ZIP to 'whisper-deepfake-detection-8bit-extracted'...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_model_path)
    print("[DEBUG] Extraction complete.")

if os.path.exists(nested_path):
    model_path = nested_path
    print(f"[DEBUG] Found nested model path: {model_path}")
else:
    model_path = extracted_model_path
    print(f"[DEBUG] Using extracted model from: {model_path}")

# --- Load model and feature extractor ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WhisperForAudioClassification.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto"
).eval()

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

print(f"[DEBUG] Model loaded to: {model.device}")
print(f"[DEBUG] Model type of encoder fc1: {type(model.encoder.layers[0].fc1)}")


id2label = {0: "fake", 1: "real"}

# --- Preprocessing ---
def preprocess_audio(file_path, sampling_rate=16000, max_duration=10.0):
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    print(f"[DEBUG] Audio shape: {audio.shape}, dtype: {audio.dtype}")
    print(f"[DEBUG] Audio mean: {audio.mean().item():.4f}, std: {audio.std().item():.4f}")
    print(f"[DEBUG] First 10 samples: {audio[:10]}")


    max_len = int(sampling_rate * max_duration)

    if len(audio) > max_len:
        audio = audio[:max_len]
    elif len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))

    features = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    return features

# --- Inference ---
@GPU
def infer(file_path):
    print("[DEBUG] infer() function triggered")
    inputs = preprocess_audio(file_path)
    inputs = {k: v.to(model.device).half() for k, v in inputs.items()}

    for k, v in inputs.items():
        print(f"[DEBUG] Input: {k} — dtype: {v.dtype}, device: {v.device}")

    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    label = id2label[pred_id].capitalize()
    confidence = probs[pred_id]
    
    print(f"[DEBUG] Logits: {logits.cpu().numpy()}")
    print(f"[DEBUG] Softmax probs: {probs}")
    print(f"[DEBUG] Predicted label: {label}, confidence: {confidence:.4f}")

    return label, confidence

if __name__ == "__main__":
    audio_folder = os.path.join(os.path.dirname(__file__), "test_audios")
    print("🔍 Running inference on audio files in test_audios...\n")

    for filename in os.listdir(audio_folder):
        if not filename.lower().endswith((".mp3", ".wav", ".flac")):
            continue

        file_path = os.path.join(audio_folder, filename)

        try:
            pred_label, conf = infer(file_path)
            print(f"{filename:25s} | Predicted: {pred_label.upper():5s} | Confidence: {conf * 100:.2f}%")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
