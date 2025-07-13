import json
import torch
import re
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, Seq2SeqTrainer
import os
from typing import List, Dict, Any

# --- 1. Serialization ---

def serialize_declarations(declarations: List[Dict[str, Any]]) -> str:
    """Converts a list of declaration dictionaries into a compact JSON string."""
    return json.dumps(declarations, ensure_ascii=False, separators=(',', ':'))

# --- 2. PyTorch Dataset ---

class DeclarativeNetworkDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: T5Tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        print(f"Loading and processing data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                text = sample.get("text")
                declarations = sample.get("declarations")
                if not text or not declarations: continue

                self.inputs.append(f"describe state: {text}")
                self.targets.append(serialize_declarations(declarations))
        print(f"Loaded {len(self.inputs)} samples.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text, target_text = self.inputs[idx], self.targets[idx]
        input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        labels = target_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {'input_ids': input_encoding['input_ids'].flatten(), 'attention_mask': input_encoding['attention_mask'].flatten(), 'labels': labels.flatten()}

# --- 3. Main Training Function ---

def train():
    MODEL_NAME = 't5-small'
    # Updated to use the new procedural data file
    DATA_PATH = 'procedural_training_data.jsonl'
    OUTPUT_DIR = './models/declarative_command_network_t5'

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Training data not found. Run `procedural_generator.py` first.")
        return

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    dataset = DeclarativeNetworkDataset(DATA_PATH, tokenizer)

    if len(dataset) < 2:
        print("ERROR: Not enough data to create a train/eval split. Need at least 2 samples.")
        return

    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        load_best_model_at_end=True,
    )

    def compute_metrics(p):
        preds, labels = p
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        exact_matches = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred.strip() == label.strip())
        return {"exact_match": exact_matches / len(decoded_preds)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training on procedurally generated data...")
    trainer.train()
    print("Training finished.")

    print(f"Saving best model and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("Model and tokenizer saved.")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure all dependencies like 'torch', 'transformers', 'sentencepiece' are installed and that you have enough system resources.")
