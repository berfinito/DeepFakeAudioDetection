import os
import librosa
import hashlib
import pandas as pd
from tqdm import tqdm

real_folder = "./real"
fake_folder = "./fake"
output_csv = "duplicates.csv"
sample_rate = 16000

# SHA-256 hash for audio waveform after normalization
def get_audio_hash(file_path, sr=sample_rate):
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y)
        return hashlib.sha256(y.tobytes()).hexdigest()
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

def find_audio_duplicates(real_folder, fake_folder, output_csv):
    hashes = {}
    duplicates = []

    # Collect all real audio files
    real_files = [os.path.join(root, file_name)
                  for root, _, files in os.walk(real_folder)
                  for file_name in files if file_name.endswith(('.flac', '.wav', '.mp3'))]

    # Process real folder and store hashes
    print("Processing real folder...")
    for file_path in tqdm(real_files, desc="Hashing real audio"):
        audio_hash = get_audio_hash(file_path)
        if audio_hash:
            hashes[audio_hash] = file_path

    # Collect all fake audio files
    fake_files = [os.path.join(root, file_name)
                  for root, _, files in os.walk(fake_folder)
                  for file_name in files if file_name.endswith(('.flac', '.wav', '.mp3'))]

    # Process fake folder and check for duplicates
    print("Processing fake folder...")
    for file_path in tqdm(fake_files, desc="Checking duplicates"):
        audio_hash = get_audio_hash(file_path)
        if audio_hash and audio_hash in hashes:
            duplicates.append({
                "Original Path": hashes[audio_hash],
                "Duplicate Path": file_path
            })

    # Save duplicates to CSV
    if duplicates:
        df = pd.DataFrame(duplicates)
        df.to_csv(output_csv, index=False)
        print(f"Duplicates found and saved to {output_csv}")
    else:
        print("No duplicates found.")

find_audio_duplicates(real_folder, fake_folder, output_csv)

print("Duplicate detection completed.")
