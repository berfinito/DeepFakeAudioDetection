import os
import subprocess
from tqdm import tqdm

real_folder = "./real"
fake_folder = "./fake"

def get_audio_duration(file_path):
    """Get the duration of an audio file using ffmpeg."""
    try:
        result = subprocess.run(
            ["ffprobe", "-i", file_path, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return 0.0

def calculate_total_duration(folder):
    """Calculate total audio duration for all files."""
    total_duration = 0.0
    audio_files = [os.path.join(root, file)
                   for root, _, files in os.walk(folder)
                   for file in files if file.endswith(('.flac', '.wav', '.mp3'))]

    for file_path in tqdm(audio_files, desc=f"Calculating duration for {folder}"):
        total_duration += get_audio_duration(file_path)

    return total_duration

real_duration = calculate_total_duration(real_folder)
fake_duration = calculate_total_duration(fake_folder)

print("\n=== Total Audio Duration ===")
print(f"Real folder: {real_duration:.2f} seconds ({real_duration / 3600:.2f} hours)")
print(f"Fake folder: {fake_duration:.2f} seconds ({fake_duration / 3600:.2f} hours)")
