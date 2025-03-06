import librosa
import numpy as np
import os
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Dataset import Audio_Dataset

# Function to process audio file and extract Mel spectrogram
def process_audio(file_path):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Extract Mel spectrogram
        # mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)

        if len(audio.shape) == 2:  # If stereo
            audio = np.mean(audio, axis=0)
        
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)

        # Convert to dB scale
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return mel_spectrogram_db
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to process the dataset using ThreadPoolExecutor
def process_dataset(input_dir):
    # List all files in the directory
    files = os.listdir(input_dir)
    files = [os.path.join(input_dir, file) for file in files if file.endswith('.wav')]  # Adjust file extension if needed

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Use tqdm to show progress while processing
        results = list(tqdm(executor.map(process_audio, files), total=len(files), desc="Processing Audio Files"))
    
    # Remove None values (in case of errors) from results
    results = [result for result in results if result is not None]
    
    return results

# Define the input directory
input_dir_fake = "dataset_equal_count\\dataset_equal_count\\fake"

# Process the dataset
processed_data_fake = process_dataset(input_dir_fake)

input_dir_real = "dataset_equal_count\\dataset_equal_count\\real"

processed_data_real = process_dataset(input_dir_real)

# Show the processed data (Mel spectrograms)
print(f"Processed {len(processed_data_fake) + len(processed_data_real)} audio files.")


df_real = pd.DataFrame()

df_real["audio_arrays"] = processed_data_real
df_real["label"] = 0

df_fake = pd.DataFrame()
df_fake["audio_arrays"] = processed_data_fake
df_fake["label"] = 1

fdf = pd.concat([df_fake, df_real], ignore_index=True)


# Function to apply Min-Max Normalization
def min_max_normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

# Apply normalization to each row in the column
fdf["audio_arrays"] = fdf["audio_arrays"].apply(lambda x: min_max_normalize(x) if isinstance(x, np.ndarray) else x)


# Split DataFrame into training and testing sets (80-20 split)
train_df, test_df = train_test_split(fdf, test_size=0.2, random_state=42)

# Create Dataset objects
train_dataset = Audio_Dataset(train_df)
test_dataset = Audio_Dataset(test_df)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
