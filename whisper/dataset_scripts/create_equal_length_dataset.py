import os
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# As we already created equal_count dataset,
# this code removes the excess length on the real part of that dataset
# to create equal length dataset

# Folder containing the longer part of dataset
real_folder = "./dataset_equal_length/real"

# Target duration to match the fake dataset
target_duration = 4773.0  # in seconds

max_workers = 24

def get_audio_duration(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return 0.0

# Collect all real audio files
def collect_real_files(folder):
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(('.flac', '.wav', '.mp3'))]

# Calculate total duration
def calculate_total_duration(file_list):
    total_duration = 0.0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_audio_duration, file): file for file in file_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating total duration"):
            total_duration += future.result()
    return total_duration

# Trim the dataset by removing shortest files
def trim_real_dataset(files, target_duration):
    current_duration = calculate_total_duration(files)
    excess_duration = current_duration - target_duration

    if excess_duration <= 0:
        print("No trimming needed. Dataset already within target duration.")
        return []

    print(f"Excess duration to remove: {excess_duration:.2f} seconds.")

    # Get file durations
    file_durations = [(file, get_audio_duration(file)) for file in files]
    file_durations.sort(key=lambda x: x[1])  # Shortest first

    # Remove shortest files until excess is trimmed
    trimmed_files = []
    trimmed_duration = 0.0

    for file, duration in file_durations:
        if trimmed_duration + duration <= excess_duration:
            os.remove(file)
            trimmed_duration += duration
            trimmed_files.append(file)
        if trimmed_duration >= excess_duration:
            break

    print(f"Trimmed {trimmed_duration:.2f} seconds by removing {len(trimmed_files)} files.")
    return trimmed_files

if __name__ == "__main__":
    real_files = collect_real_files(real_folder)
    print(f"Found {len(real_files)} real audio files.")

    # Trim dataset
    trimmed_files = trim_real_dataset(real_files, target_duration)

    # Recalculate final duration
    final_duration = calculate_total_duration(collect_real_files(real_folder))

    print(f"Final duration of real dataset: {final_duration:.2f} seconds ({final_duration / 3600:.2f} hours)")
    print(f"Trimmed {len(trimmed_files)} files. Dataset balancing completed successfully.")
