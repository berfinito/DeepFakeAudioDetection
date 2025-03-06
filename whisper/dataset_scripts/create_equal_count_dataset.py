import os
import random
import shutil
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

real_folder = "./real"
fake_folder = "./fake"

# Output folders
output_equal_count = "./dataset_equal_count"
os.makedirs(output_equal_count, exist_ok=True)
os.makedirs(f"{output_equal_count}/real", exist_ok=True)
os.makedirs(f"{output_equal_count}/fake", exist_ok=True)

def get_dataset_name(file_name):
    return file_name.split("_")[0]

# collect files by dataset
def collect_files(folder):
    dataset_files = defaultdict(list)
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('.flac', '.wav', '.mp3')):
                dataset_name = get_dataset_name(file)
                dataset_files[dataset_name].append(os.path.join(root, file))
    return dataset_files

# select files by duration while balancing datasets
def select_files_by_duration(files_dict, target_duration):
    selected_files = []
    total_duration = 0.0
    dataset_durations = defaultdict(float)

    for dataset, files in files_dict.items():
        random.shuffle(files)
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = {executor.submit(get_audio_duration, file): file for file in files}
            for future in as_completed(futures):
                file_path = futures[future]
                duration = future.result()
                if total_duration + duration <= target_duration:
                    selected_files.append(file_path)
                    total_duration += duration
                    dataset_durations[dataset] += duration
                if total_duration >= target_duration:
                    break

    # Balance dataset contribution
    min_duration = min(dataset_durations.values(), default=0)
    balanced_files = [f for f in selected_files if dataset_durations[get_dataset_name(os.path.basename(f))] >= min_duration]

    return balanced_files

real_files = collect_files(real_folder)
fake_files = collect_files(fake_folder)

# Function to randomly select balanced files
def select_balanced_files(files_dict, target_count):
    selected_files = []
    per_dataset_count = target_count // len(files_dict)
    for dataset, files in files_dict.items():
        selected_files.extend(random.sample(files, min(per_dataset_count, len(files))))
    return selected_files

max_workers = 24

def get_audio_duration(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return 0.0

# Calculate total duration with threading
def calculate_total_duration(file_list):
    total_duration = 0.0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_audio_duration, file): file for file in file_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating total duration"):
            total_duration += future.result()
    return total_duration

# Step 1: Create Dataset with Equal Counts (1200 real, 1200 fake)
print("Creating Dataset 1: Equal Count (1200 real, 1200 fake)...")
selected_real = select_balanced_files(real_files, 1200)
selected_fake = select_balanced_files(fake_files, 1200)

# Calculate total duration for 1200 real and fake datasets
real_duration = calculate_total_duration(selected_real)
fake_duration = calculate_total_duration(selected_fake)

print(f"Total duration of selected 1200 real files: {real_duration:.2f} seconds ({real_duration / 3600:.2f} hours)")
print(f"Total duration of selected 1200 fake files: {fake_duration:.2f} seconds ({fake_duration / 3600:.2f} hours)")

# Copy files for equal count dataset
for file_path in tqdm(selected_real, desc="Copying real (equal count)"):
    shutil.copy2(file_path, f"{output_equal_count}/real")

for file_path in tqdm(selected_fake, desc="Copying fake (equal count)"):
    shutil.copy2(file_path, f"{output_equal_count}/fake")

