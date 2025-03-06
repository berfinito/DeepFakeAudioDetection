import torch
import whisper
import torchaudio
import torchaudio.transforms as at
from transformers import WhisperModel
import torch.nn as nn

# Same class with training
class WhisperDeepFakeClassifier(nn.Module):
    def __init__(self, model_name="openai/whisper-tiny", num_classes=2):
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

# Load model and set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WhisperDeepFakeClassifier().to(device)
model.load_state_dict(torch.load("whisper_deepfake_model.pth", map_location=device))
model.eval()

# Run inference on an audio file
def infer(audio_path):
    input_features = preprocess_audio(audio_path).to(device)
    with torch.no_grad():
        output = model(input_features)
        predicted_label = torch.argmax(output, dim=1).item()
    
    label_map = {0: "Real", 1: "Fake"}
    print(f"File: {audio_path}, Prediction: {label_map[predicted_label]}")
    return label_map[predicted_label]

audio_file1 = "test_audio_real.mp3"
audio_file2 = "test_audio_fake.wav"
infer(audio_file1)
infer(audio_file2)
