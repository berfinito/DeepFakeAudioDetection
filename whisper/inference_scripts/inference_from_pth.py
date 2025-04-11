import torch
import whisper
import torchaudio
import torchaudio.transforms as at
from transformers import WhisperModel
import torch.nn as nn
import os
from huggingface_hub import hf_hub_download
import torch.nn.functional as F

SKIP_MODEL_LOADING = False # Flag for testing purposes

MODEL_FILENAME = "whisper_deepfake_model.pth"
MODEL_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "models", MODEL_FILENAME)

# Same class with training
class WhisperDeepFakeClassifier(nn.Module):
    def __init__(self, model_name="openai/whisper-small", num_classes=2):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = self.whisper.encoder
        self.freeze_encoder()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, input_features):
        encoder_outputs = self.encoder(input_features=input_features).last_hidden_state
        pooled_output = encoder_outputs.mean(dim=1)
        return self.classifier(pooled_output)

# Function to load and preprocess an audio file
def preprocess_audio(audio_path, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path, normalize=True)
    if sr != sample_rate:
        waveform = at.Resample(sr, sample_rate)(waveform)
    
    # Whisper expects a Mel spectrogram
    audio = whisper.pad_or_trim(waveform.flatten())  # Ensure length consistency
    mel = whisper.log_mel_spectrogram(audio)  # Convert to Mel spectrogram
    return mel.unsqueeze(0)  # Add batch dimension

# Function to load model and set it to evaluation mode
def load_model():
    model = WhisperDeepFakeClassifier().to(device)

    if SKIP_MODEL_LOADING:
        print("Skipping model loading.")
        return model

    if not os.path.exists(MODEL_LOCAL_PATH):
        print("Local model not found. Downloading from Hugging Face Model Hub...")
        downloaded_path = hf_hub_download(
            repo_id="refikbklm/fake-audio-detector-model",
            filename=MODEL_FILENAME
        )
        torch.save(torch.load(downloaded_path, map_location=device), MODEL_LOCAL_PATH)

    print(f"Loading model from {MODEL_LOCAL_PATH}...")
    model.load_state_dict(torch.load(MODEL_LOCAL_PATH, map_location=device))
    model.eval()
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model()

# Run inference on an audio file
def infer(audio_path):
    if SKIP_MODEL_LOADING:
        print(f"[DEV MODE] Stub prediction for: {audio_path}")
        return "Real (stubbed)"
    
    input_features = preprocess_audio(audio_path).to(device)
    with torch.no_grad():
        output = model(input_features)
        probs = F.softmax(output, dim=1)
        predicted_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_label].item()

    label_map = {0: "Real", 1: "Fake"}
    label = label_map[predicted_label]
    print(f"File: {audio_path}, Prediction: {label} ({confidence * 100:.2f}%)")
    return label, confidence

if __name__ == "__main__":
    audio_folder = os.path.join(os.path.dirname(__file__), "test_audios")
    supported_formats = (".mp3", ".wav", ".flac")

    for filename in os.listdir(audio_folder):
        if filename.lower().endswith(supported_formats):
            audio_path = os.path.join(audio_folder, filename)
            infer(audio_path)