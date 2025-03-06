import os
import whisper
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from transformers import WhisperModel
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as at
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing

# Optimize CUDA performance by disabling benchmarking (as dataset has different audio lengths) and clearing cache
torch.backends.cudnn.benchmark = False  
torch.cuda.empty_cache()

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    """
    Load an audio file and resample it to a target sample rate if needed.
    """
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

class AudioDataset(Dataset):
    """
    Custom PyTorch dataset for loading and processing audio samples.
    """
    def __init__(self, audio_info_list, labels, sample_rate) -> None:
        super().__init__()
        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.labels = labels

    def __len__(self):
        return len(self.audio_info_list)
    
    def __getitem__(self, idx):
        """
        Retrieve an audio sample and its corresponding label.
        """
        audio_path = self.audio_info_list[idx]
        label = self.labels[idx]

        # Load and preprocess audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten()) # Ensure consistent input length
        mel = whisper.log_mel_spectrogram(audio)  # Convert audio to Mel spectrogram

        return {
            "input_ids": mel,
            "labels": torch.tensor(label, dtype=torch.long)
        }

class WhisperDeepFakeClassifier(nn.Module):
    """
    Neural network using Whisper encoder for deepfake audio classification.
    """
    def __init__(self, model_name="openai/whisper-tiny", num_classes=2):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = self.whisper.encoder # Use only the encoder for feature extraction
        self.freeze_encoder()
        
        # Classification head for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def freeze_encoder(self):
        """
        Freeze encoder weights to prevent updates during training.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, input_features):
        """
        Extract features using Whisper encoder and classify them.
        """
        encoder_outputs = self.encoder(input_features=input_features).last_hidden_state
        pooled_output = encoder_outputs.mean(dim=1)  # Global average pooling
        return self.classifier(pooled_output)

def train(model, dataloader, criterion, optimizer, device, scaler):
    """
    Train the model for one epoch using mixed precision training.
    """
    model.train()
    total_loss, correct = 0, 0

    for batch in tqdm(dataloader, desc='Training', leave=False, dynamic_ncols=True):
        features = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(features)
            loss = criterion(outputs, labels)

        # Backpropagation with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)

    return avg_loss, accuracy

def evaluate(model, eval_loader, criterion, device):
    """
    Evaluate the model performance and generate confusion matrix and precision-recall curve.
    """
    model.eval()
    total_loss, correct = 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluation', leave=False, dynamic_ncols=True):
            features = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(features)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(eval_loader)
    accuracy = correct / len(eval_loader.dataset)
    print(f"Eval Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Compute and plot confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    #plt.show()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    # Compute and plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig("precision_recall_curve.png") 
    plt.close()

    early_stop_scheduler.step(avg_loss)  # Reduce LR if loss stops improving
    return avg_loss, accuracy

if __name__ == "__main__":
    run_training = True 
    multiprocessing.set_start_method('spawn', force=True)  # Prevent script from running twice on Dataloader workers

    fake_dir_path = "audio_dataset/fake"
    real_dir_path = "audio_dataset/real"

    fake_files = os.listdir(fake_dir_path)
    real_files = os.listdir(real_dir_path)
    fake_dict = {os.path.join(fake_dir_path, fake): 1 for fake in fake_files}
    real_dict = {os.path.join(real_dir_path, real): 0 for real in real_files}

    file_paths = list(real_dict.keys()) + list(fake_dict.keys())
    labels = list(real_dict.values()) + list(fake_dict.values())
    print(f"Total files: {len(file_paths)}, Total labels: {len(labels)}")

    # Setup device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset and DataLoaders
    dataset = AudioDataset(file_paths, labels=labels, sample_rate=16000)
    torch.manual_seed(42) # Set seed to get same split everytime
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    eval_size = dataset_size - train_size

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=448, shuffle=True, drop_last=True, num_workers=10, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    eval_loader = DataLoader(eval_dataset, batch_size=384, shuffle=False, drop_last=True, num_workers=6, pin_memory=True, prefetch_factor=2)

    model = WhisperDeepFakeClassifier().to(device)
    print(f"Model is running on: {next(model.parameters()).device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    early_stop_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler(device="cuda")

    if run_training:
        # Training loop
        num_epochs = 15
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, scaler)
            scheduler.step()
            print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        print("Training Complete! Saving model.")
        torch.save(model.state_dict(), "whisper_deepfake_model.pth")
    
    # Evaulation
    model = WhisperDeepFakeClassifier().to(device)
    model.load_state_dict(torch.load("whisper_deepfake_model.pth"))
    model.eval()  # Set model to evaluation mode
    print("Running final evaluation...")
    eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)
    print("Evaluation complete.")