import os
import librosa
from tqdm import tqdm

real_folder = "./real"
fake_folder = "./fake"
sample_rate = 16000 

def calculate_total_duration(folder):
    """Calculate the total duration of all audio files."""
    total_duration = 0.0
    audio_files = [os.path.join(root, file)
                   for root, _, files in os.walk(folder)
                   for file in files if file.endswith(('.flac', '.wav', '.mp3'))]

    for file_path in tqdm(audio_files, desc=f"Calculating duration for {folder}"):
        try:
            y, sr = librosa.load(file_path, sr=sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            total_duration += duration
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    return total_duration

real_duration = calculate_total_duration(real_folder)
fake_duration = calculate_total_duration(fake_folder)

print("\n=== Total Audio Duration ===")
print(f"Real folder: {real_duration:.2f} seconds ({real_duration / 3600:.2f} hours)")
print(f"Fake folder: {fake_duration:.2f} seconds ({fake_duration / 3600:.2f} hours)")
