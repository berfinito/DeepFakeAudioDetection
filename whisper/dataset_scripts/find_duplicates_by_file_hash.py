import os
import hashlib
import csv
from multiprocessing import Pool, Manager
from tqdm import tqdm

folders = ["real", "fake"]
output_csv = "duplicates.csv"

# SHA256 hash of file
def get_file_hash(file_path):
    try:
        with open(file_path, "rb") as f:
            hasher = hashlib.sha256()
            while chunk := f.read(8192):
                hasher.update(chunk)
        return file_path, hasher.hexdigest()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, None

# Function to collect all audio files
def collect_audio_files(folder):
    audio_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
    return audio_files

# Function to find duplicates and save to CSV
def find_duplicates():
    all_audio_files = []
    for folder in folders:
        if os.path.exists(folder):
            all_audio_files.extend(collect_audio_files(folder))

    print(f"Found {len(all_audio_files)} audio files. Calculating hashes...")

    with Pool() as pool:
        results = list(tqdm(pool.imap(get_file_hash, all_audio_files), total=len(all_audio_files)))

    hash_dict = {}
    duplicates = []

    for file_path, file_hash in results:
        if file_hash:
            if file_hash in hash_dict:
                original = hash_dict[file_hash]
                duplicates.append((original, file_path))
            else:
                hash_dict[file_hash] = file_path

    # Save results to CSV
    if duplicates:
        print(f"Found {len(duplicates)} duplicate files. Saving to {output_csv}...")
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Original Path", "Duplicate Path"])
            for original, duplicate in duplicates:
                writer.writerow([original, duplicate])
        print(f"Duplicates saved to {output_csv}")
    else:
        print("No duplicate audio files found.")

if __name__ == "__main__":
    find_duplicates()
