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

The model weights file is **large (1GB)**, so it is stored using **Git LFS (Large File Storage)**.

### Download Options:

### 🔹 Automatic Download
If `whisper_deepfake_model.pth` is missing, the script will attempt to download it from Hugging Face.

### 🔹 Manual Download
You can also [download it manually](https://drive.google.com/file/d/1HXai8S5tKczl6Z9uXynQqekpSrrQrxrL/view?usp=sharing) and place it in:

```
inference_scripts/models/whisper_deepfake_model.pth
```

### 🔹 Git LFS Alternative
```bash
git lfs install
git lfs pull
```

## Whisper Model Training and Inference

### **Dataset Format for Training**
To train the model, ensure your dataset follows the format below and update the path as needed inside the training scripts.

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
All training scripts are located in `training_scripts/`:

- `train_bnb.py` — 8-bit quantized training with `bitsandbytes`
- `train_lora.py` — LoRA fine-tuning
- `train_lora_and_bnb.py` — LoRA + 8-bit combined
- `train_custom_classifier.py` — Custom classification head only

---

### **Inference**
Choose one of the following scripts under `inference_scripts/`:

### 1. `inference_from_pth.py`  
Full-precision model inference using a `.pth` file.  
Requires `whisper_deepfake_model.pth` in the `models/` folder.  
The script will automatically download it if not found.

```bash
cd inference_scripts
python inference_from_pth.py
```

### 2. `inference_from_tensor.py`  
Lightweight 8-bit inference using `safetensors`.  
Looks for a quantized model inside the `models/` directory.

```bash
cd inference_scripts
python inference_from_tensor.py
```

> Place your `.wav`, `.mp3`, or `.flac` files in `test_audios/` before running inference.
