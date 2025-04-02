# Deepfake Audio Detection

## Website

You can test the model directly in the [Hugging Face Space](https://huggingface.co/spaces/refikbklm/fake-audio-detector).

## Dependencies
Run the following commands to install the necessary dependencies:

```
pip install git+https://github.com/openai/whisper.git
```
```
pip install -r requirements.txt
```

## Handling Large Files (Git LFS)

The model weights file is **large (144MB)**, so it is stored using **Git LFS (Large File Storage)**.

By default, **Git LFS should automatically download large files** when you clone the repository.  
However, if the file **did not download** or appears **smaller than expected**, follow one of these methods:

### **Option 1: Download from Google Drive**
You can manually download the model weights from **[Google Drive](https://drive.google.com/file/d/13OUJ9D5oG3K4ci_CZpMWbLja5yApgWg0/view)** and place it inside the `whisper/` folder.

### **Option 2: Get It Using Git LFS**
If you prefer to fetch the model via Git LFS, follow these steps:

1️⃣ **Install Git LFS** (**if not already installed**):

   - **Windows**: Download and install from [git-lfs.github.com](https://git-lfs.github.com/)  
   - **Mac (Homebrew)**:  
     ```
     brew install git-lfs
     ```
   - **Linux**:  
     ```
     sudo apt install git-lfs  # Debian/Ubuntu
     sudo dnf install git-lfs  # Fedora
     ```

2️⃣ **After cloning the repository, run:**
   ```
   git lfs install
   git lfs pull
   ```

This will **download all LFS-tracked files**, including `whisper_deepfake_model.pth`.

---

## Whisper Model

### **Dataset Format for Training**
To train the model, ensure your `audio_dataset` folder (containing `fake/` and `real/` subfolders) is placed **next to** `train_eval.py`.  
The dataset should be structured as follows:

```
audio_dataset/
│── fake/
│   ├── fake1.wav
│   ├── fake2.mp3
│   ├── ...
│
│── real/
│   ├── real1.wav
│   ├── real2.mp3
│   ├── ...
```

### **Training & Evaluation**
Before training, **adjust `batch_size` and `num_workers`** in the `DataLoaders` inside `train_eval.py` to match your system's specifications.  

Then, start the training process by running:
```
py train_eval.py
```
This will **train the model, save the trained weights, and evaluate its performance**.

---

### **Inference**
To run inference, ensure that the **script, model weights (`.pth`), and test audio files** are all in the same folder:

```
./
│── inference.py
│── whisper_deepfake_model.pth
│── test_audio_real.mp3
│── test_audio_fake.wav
```

Run the following command:
```
py inference.py
```
This will **process the test files and print their predicted labels**.
