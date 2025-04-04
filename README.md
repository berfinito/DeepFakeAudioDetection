# Deepfake Audio Detection

## Website

You can test the model directly in the [Hugging Face Space](https://huggingface.co/spaces/refikbklm/fake-audio-detector).

## Dependencies
Run the following commands to install the necessary dependencies:

```bash
pip install git+https://github.com/openai/whisper.git
```
```bash
pip install -r requirements.txt
```

## Model Weights for Inference

The model weights file is **large (3GB)**, so it is stored using **Git LFS (Large File Storage)**.

### How it's handled:

- If you have **Git LFS installed**, the model file (`whisper_deepfake_model.pth`) will likely be **downloaded automatically** when you clone the repository.
- If you **don’t have Git LFS**, the file will appear as a small pointer text file — and the code will **automatically download the full model from Hugging Face** when needed.

So in most cases, **you don’t need to do anything manually**.

If you still prefer to download the model manually, you can use one of these options:

### 📥 **Option 1: Download from Google Drive**
If you prefer, you can manually download the model weights from **[Google Drive](https://drive.google.com/file/d/13OUJ9D5oG3K4ci_CZpMWbLja5yApgWg0/view)** and place it in the same folder as `inference.py`

### 💻 **Option 2: Get It Using Git LFS**
If you prefer to fetch the model via Git LFS, follow these steps:

1️⃣ **Install Git LFS** (**if not already installed**):

   - **Windows**: Download and install from [git-lfs.github.com](https://git-lfs.github.com/)  
   - **Mac (Homebrew)**:  
     ```bash
     brew install git-lfs
     ```
   - **Linux**:  
     ```bash
     sudo apt install git-lfs  # Debian/Ubuntu
     sudo dnf install git-lfs  # Fedora
     ```

2️⃣ **After cloning the repository, run:**
   ```bash
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
```bash
py train_eval.py
```
This will **train the model, save the trained weights, and evaluate its performance**.

---

### **Inference**
To run inference, make sure the following are in the **same directory**:

```
./
│── inference.py
│── whisper_deepfake_model.pth  # will be auto-downloaded if missing
│── test_audio_real.mp3
│── test_audio_fake.wav
```

Then run the following command:
```bash
py inference.py
```
This will **process the test files and print their predicted labels**.
