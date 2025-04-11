import torch
import os
import json
import random
import librosa
import numpy as np
import logging
import matplotlib.pyplot as plt
from datasets import load_from_disk, concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperForAudioClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# ---- Config ----
PROCESS_DATASET = False
USE_4BIT = True  # Set to True to use 4-bit quantization (QLoRA-style)
model_name = "openai/whisper-medium"

# ---- Setup ----
logging.basicConfig(level=logging.INFO)
set_seed(42)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

label_list = ["fake", "real"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# ---- Preprocessing ----
def preprocess_function(examples):
    max_length = 16000 * 10  # 10 seconds

    def load_audio(file_path):
        audio, sr = librosa.load(file_path, sr=16000)
        if len(audio) > max_length:
            audio = audio[:max_length]
        elif len(audio) < max_length:
            padding = np.zeros(max_length - len(audio))
            audio = np.concatenate((audio, padding))
        return audio

    audio_arrays = [load_audio(p) for p in examples["audio"]]
    inputs = feature_extractor(audio_arrays, sampling_rate=16000, return_tensors="pt")
    inputs["labels"] = [label2id[label] for label in examples["label"]]
    return inputs

# ---- Model prep: Quantization + LoRA ----
def prepare_model():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        load_in_8bit=not USE_4BIT,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
    )

    model = WhisperForAudioClassification.from_pretrained(
        model_name,
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
        quantization_config=quant_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ---- Metrics ----
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ---- Main ----
if __name__ == "__main__":
    fake_dir = os.listdir("dataset/fake60k")
    real_dir = os.listdir("dataset/real60k")

    fake_dict = {f"dataset/fake60k/{f}": "fake" for f in fake_dir}
    real_dict = {f"dataset/real60k/{f}": "real" for f in real_dir}

    audio_paths = list(real_dict.keys()) + list(fake_dict.keys())
    labels = list(real_dict.values()) + list(fake_dict.values())

    if PROCESS_DATASET:
        audio_dataset = Dataset.from_dict({"audio": audio_paths, "label": labels})
        train_idx, eval_idx = train_test_split(range(len(audio_dataset)), test_size=0.1, random_state=42)
        train_dataset = audio_dataset.select(train_idx)
        eval_dataset = audio_dataset.select(eval_idx)

        def process_chunks(dataset, tag):
            chunks = []
            for i in range(0, len(dataset), 2000):
                chunk = dataset.select(range(i, min(i + 2000, len(dataset))))
                processed = chunk.map(
                    preprocess_function, batched=True, batch_size=100,
                    num_proc=2, load_from_cache_file=False
                )
                path = f"processed_{tag}_chunks/chunk_{i//2000}"
                processed.save_to_disk(path)
                chunks.append(load_from_disk(path))
            return concatenate_datasets(chunks)

        train_dataset = process_chunks(train_dataset, "train")
        eval_dataset = process_chunks(eval_dataset, "eval")

        train_dataset.save_to_disk("processed_train_dataset")
        eval_dataset.save_to_disk("processed_eval_dataset")
        exit()
    else:
        train_dataset = load_from_disk("dataset/processed_train_dataset")
        eval_dataset = load_from_disk("dataset/processed_eval_dataset")

    for ds in (train_dataset, eval_dataset):
        ds.remove_columns(["label", "audio"])
        ds.set_format(type="torch", columns=["input_features", "labels"])

    model = prepare_model()

    training_args = TrainingArguments(
        output_dir="./results_lora_quant",
        num_train_epochs=5,
        per_device_train_batch_size=36,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs_lora_quant",
        logging_steps=25,
        save_total_limit=2,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        optim="paged_adamw_32bit",
        dataloader_num_workers=4,
        label_names=["labels"],
        greater_is_better=False, 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)]
    )

    trainer.train(resume_from_checkpoint="./results_lora_quant/checkpoint-3114")

    model.save_pretrained("whisper-lora-quantized", safe_serialization=True)
    processor.save_pretrained("whisper-lora-quantized")

    merged = PeftModel.from_pretrained(model, "whisper-lora-quantized").merge_and_unload()
    merged.save_pretrained("whisper-lora-merged")

    preds = trainer.predict(eval_dataset)
    ConfusionMatrixDisplay.from_predictions(preds.label_ids, preds.predictions.argmax(-1), display_labels=["REAL", "FAKE"])
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_lora_quant.png")
    plt.close()

    metrics = compute_metrics(preds)
    with open("final_metrics_lora_quant.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Training complete and saved.")
