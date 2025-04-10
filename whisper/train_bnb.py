import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification, Trainer, TrainingArguments, WhisperProcessor, AutoModelForAudioClassification, BitsAndBytesConfig, WhisperModel, set_seed
import numpy as np
import librosa
import os
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from torch.optim import AdamW
import matplotlib.pyplot as plt
import json
from datasets import Dataset, Audio
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

PROCESS_DATASET = False

logging.basicConfig(level=logging.INFO)
set_seed(42)
model_name="openai/whisper-small"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

label_list = ["fake", "real"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

def preprocess_function(examples):
    max_length = 16000 * 10
    
    def load_audio(file_path):
        audio, sr = librosa.load(file_path, sr=16000)
        if len(audio) > max_length:
            audio = audio[:max_length]
        elif len(audio) < max_length:
            # Pad audio if it's shorter than 10 seconds
            padding = np.zeros(max_length - len(audio))
            audio = np.concatenate((audio, padding))
        return audio

    audio_arrays = [load_audio(file_path) for file_path in examples["audio"]]
    
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        return_tensors="pt"
    )
    inputs["labels"] = [label2id[label] for label in examples["label"]]
    return inputs

def prepare_whisper_classifier_with_8bit_hybrid(
    model_name="openai/whisper-small",
    num_labels=2,
    label2id=None,
    id2label=None,
    freeze_base_model=True
):
    """
    Loads a Whisper classification model with the encoder quantized to 8-bit.
    Keeps Hugging Face's classifier head in full precision.
    Follows the original freezing logic (freeze all, unfreeze classifier).
    """
    # Define 8-bit quantization config for encoder
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4"
    )

    # Load encoder in 8-bit
    whisper_encoder = WhisperModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    ).encoder

    # Load classification model normally
    model = WhisperForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    # Swap in the quantized encoder
    model.encoder = whisper_encoder

    if freeze_base_model:
        # Freeze all params
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classifier head
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Print trainable stats
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"✅ Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.2%})")

    return model

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    fake_dir_path = "dataset/fake60k"
    real_dir_path = "dataset/real60k"

    # Get the list of files
    fake_dir = os.listdir(fake_dir_path)
    real_dir = os.listdir(real_dir_path)

    # Create dictionaries with labels
    fake_dict = {os.path.join(fake_dir_path, fake): "fake" for fake in fake_dir}
    real_dict = {os.path.join(real_dir_path, real): "real" for real in real_dir}

    # Prepare file paths and labels
    file_paths = list(real_dict.keys()) + list(fake_dict.keys())
    labels = list(real_dict.values()) + list(fake_dict.values())

    print(f"Total files: {len(file_paths)}, Total labels: {len(labels)}")

    bnbcongif= BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
        bnb_8bit_use_double_quant=True,  # Use double quantization for better performance
        bnb_8bit_quant_type="nf4",  # Use nf4 quantization type
    )

    # Initialize the model with 8-bit quantization
    num_labels = 2
    model = prepare_whisper_classifier_with_8bit_hybrid(
        model_name, 
        num_labels=num_labels,
        freeze_base_model=True,
        label2id=label2id,
        id2label=id2label
    )
    processor = WhisperProcessor.from_pretrained(model_name)

    # Create the combined list of file paths and labels
    audio_paths = list(real_dict.keys()) + list(fake_dict.keys())
    labels = list(real_dict.values()) + list(fake_dict.values())

    if PROCESS_DATASET:
        # Create dataset
        data_dict = {"audio": audio_paths, "label": labels}
        audio_dataset = Dataset.from_dict(data_dict)

        random.seed(42)
        indices = list(range(len(audio_dataset)))
        train_indices, eval_indices = train_test_split(indices, test_size=0.1, random_state=42)

        train_dataset = audio_dataset.select(train_indices)
        eval_dataset = audio_dataset.select(eval_indices)

        print(f"🔹 Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        chunk_size = 2000
        train_shards = []
        eval_shards = []

        # Process and save train chunks
        print("🧠 Chunked preprocessing: training set")
        for i in range(0, len(train_dataset), chunk_size):
            chunk = train_dataset.select(range(i, min(i + chunk_size, len(train_dataset))))
            print(f"📦 Processing train chunk {i // chunk_size + 1} of {len(train_dataset) // chunk_size + 1}")
            
            processed = chunk.map(
                preprocess_function,
                batched=True,
                batch_size=100,
                num_proc=2,
                load_from_cache_file=False,
                desc=f"⏳ Train chunk {i // chunk_size + 1}"
            )
            chunk_path = f"processed_train_chunks/chunk_{i//chunk_size}"
            processed.save_to_disk(chunk_path)
            train_shards.append(load_from_disk(chunk_path))  # For concatenating after

        # Combine all train chunks
        train_dataset = concatenate_datasets(train_shards)
        train_dataset.save_to_disk("processed_train_dataset")
        print("✅ Training dataset saved.")

        # Process and save eval chunks
        print("🧠 Chunked preprocessing: evaluation set")
        for i in range(0, len(eval_dataset), chunk_size):
            chunk = eval_dataset.select(range(i, min(i + chunk_size, len(eval_dataset))))
            print(f"📦 Processing eval chunk {i // chunk_size + 1} of {len(eval_dataset) // chunk_size + 1}")

            processed = chunk.map(
                preprocess_function,
                batched=True,
                batch_size=100,
                num_proc=2,
                load_from_cache_file=False,
                desc=f"⏳ Eval chunk {i // chunk_size + 1}"
            )
            chunk_path = f"processed_eval_chunks/chunk_{i//chunk_size}"
            processed.save_to_disk(chunk_path)
            eval_shards.append(load_from_disk(chunk_path))

        # Combine all eval chunks
        eval_dataset = concatenate_datasets(eval_shards)
        eval_dataset.save_to_disk("processed_eval_dataset")
        print("✅ Evaluation dataset saved.")

        exit()
    else:
        train_dataset = load_from_disk("dataset/processed_train_dataset")
        eval_dataset = load_from_disk("dataset/processed_eval_dataset")
        print(len(train_dataset))
        print(len(eval_dataset))

    train_dataset = train_dataset.remove_columns(["label"])
    eval_dataset = eval_dataset.remove_columns(["label"])
    train_dataset = train_dataset.remove_columns(["audio"])
    eval_dataset = eval_dataset.remove_columns(["audio"])

    train_dataset.set_format(type="torch", columns=["input_features", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_features", "labels"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results_whisper_8bit",
        num_train_epochs=12,
        per_device_train_batch_size=120,
        gradient_accumulation_steps=1, 
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs_whisper_8bit',
        logging_steps=25,
        save_total_limit=2,        
        save_strategy="epoch", 
        evaluation_strategy="epoch",
        fp16=True,  # Enable mixed precision training
        optim="adamw_8bit",  # Use 8-bit optimizer
        disable_tqdm=False,
        dataloader_num_workers=8,
    )


    # Create Trainer with 8-bit optimization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model - note that 8-bit models need special handling for saving
    output_dir = "./whisper-deepfake-detection-8bit"
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)

    print(f"Model trained and saved to {output_dir}")

    outputs = trainer.predict(eval_dataset)
    preds = outputs.predictions.argmax(-1)
    labels = outputs.label_ids

    ConfusionMatrixDisplay.from_predictions(labels, preds, display_labels=["REAL", "FAKE"])
    plt.title("Confusion Matrix (Final Model)")
    plt.savefig("confusion_matrix_final.png")
    plt.close()

    # Save metrics to JSON

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    with open("final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Confusion matrix and metrics saved.")