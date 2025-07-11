import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# --- Конфигурация для английского языка ---
TOKENIZER_NAME = "bert-base-uncased" # Англоязычная модель
PRETRAINED_MODEL_NAME = "bert-base-uncased"
DATA_FILE_PATH = "data/synthetic_nlu_data_en.jsonl" # Англоязычные данные
MODEL_SAVE_PATH = "models/intent_classifier_bert_en" # Путь сохранения для англ. модели

# Параметры обучения
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 128

class IntentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_idx):
        text = str(self.texts[item_idx])
        label = self.labels[item_idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False, # Для BERT не нужны token_type_ids для sequence classification
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path: str) -> Tuple[List[str], List[str]]:
    texts = []
    intent_labels_str = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    texts.append(record["text"])
                    intent_labels_str.append(record["intent"])
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line in {data_path}: {e} - Line: '{line.strip()}'")
                except KeyError as e:
                    print(f"Skipping line with missing key in {data_path}: {e} - Line: '{line.strip()}'")

    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}")
        return [], []
    return texts, intent_labels_str

def compute_metrics(pred_logits, true_labels):
    preds = np.argmax(pred_logits, axis=1)
    labels = true_labels
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    total_loss = 0

    all_logits = []
    all_true_labels = []

    for batch_idx, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu().numpy())
        all_true_labels.append(labels.detach().cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if batch_idx > 0 and batch_idx % (len(data_loader)//10 if len(data_loader) > 10 else 1) == 0: # Печатаем лог ~10 раз за эпоху
            print(f"  Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    final_logits = np.concatenate(all_logits, axis=0)
    final_true_labels = np.concatenate(all_true_labels, axis=0)
    metrics = compute_metrics(final_logits, final_true_labels)
    return metrics['accuracy'], avg_loss, metrics


def eval_model(model, data_loader, device):
    model = model.eval()
    total_loss = 0

    all_logits = []
    all_true_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu().numpy())
            all_true_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    final_logits = np.concatenate(all_logits, axis=0)
    final_true_labels = np.concatenate(all_true_labels, axis=0)
    metrics = compute_metrics(final_logits, final_true_labels)
    return metrics['accuracy'], avg_loss, metrics


def main():
    print("Starting Intent Classifier Training (English)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    texts, intent_labels_str = load_data(DATA_FILE_PATH)
    if not texts:
        return

    unique_intents = sorted(list(set(intent_labels_str)))
    intent_to_id = {intent: i for i, intent in enumerate(unique_intents)}
    id_to_intent = {i: intent for intent, i in intent_to_id.items()}
    intent_labels_ids = [intent_to_id[label] for label in intent_labels_str]
    num_classes = len(unique_intents)

    if num_classes == 0:
        print("No intent classes found. Check your data.")
        return
    if num_classes == 1 and len(texts) > 1:
        print(f"Warning: Only one intent class ('{unique_intents[0]}') found. Training might not be meaningful.")
        # sklearn.model_selection.train_test_split might fail with stratify if only one class.
        # We'll proceed but this is a data issue.
        # Consider adding more diverse intents or disabling stratify if this is intentional.

    print(f"Number of classes (intents): {num_classes}")
    print(f"Intent to ID mapping (first 5): {list(intent_to_id.items())[:5]}")

    # Stratify might fail if a class has only 1 sample, or if num_classes is 1.
    # Using a simple split if stratify fails or only one class.
    try:
        if num_classes > 1:
             train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, intent_labels_ids, test_size=0.15, random_state=42, stratify=intent_labels_ids
            )
        else: # Cannot stratify with 1 class
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, intent_labels_ids, test_size=0.15, random_state=42 # No stratify
            )
    except ValueError as e:
        print(f"Warning: train_test_split with stratify failed: {e}. Falling back to non-stratified split.")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, intent_labels_ids, test_size=0.15, random_state=42
        )

    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    if not train_texts or not val_texts:
        print("Error: Not enough data for training or validation split. Need more generated samples.")
        return

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, MAX_SEQ_LENGTH)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for main thread
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=num_classes,
        id2label=id_to_intent, # For better model introspection
        label2id=intent_to_id
    )
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * NUM_EPOCHS
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)), # Ensure at least 1 warmup step or 10%
        num_training_steps=total_steps
    )

    print("\n--- Starting Training ---")
    best_val_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_acc, train_loss, train_metrics = train_epoch(
            model, train_data_loader, optimizer, device, scheduler
        )
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        print(f"Full Train metrics: {train_metrics}")

        val_acc, val_loss, val_metrics = eval_model(
            model, val_data_loader, device
        )
        print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        print(f"Full Validation metrics: {val_metrics}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            print(f"New best validation accuracy: {best_val_accuracy:.4f}. Saving model...")
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            with open(os.path.join(MODEL_SAVE_PATH, "intent_label_map.json"), 'w', encoding='utf-8') as f:
                json.dump({"intent_to_id": intent_to_id, "id_to_intent": id_to_intent},
                          f, ensure_ascii=False, indent=4)

    print("\n--- Training Finished ---")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model, tokenizer and label map saved to {MODEL_SAVE_PATH} (if accuracy improved)")

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE_PATH):
        print(f"ERROR: Data file not found at {DATA_FILE_PATH}")
        print(f"Please run src/training_data_generator.py first to generate '{DATA_FILE_PATH}'.")
    else:
        main()
