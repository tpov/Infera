import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
import numpy as np

# --- Конфигурация для английского языка ---
TOKENIZER_NAME = "bert-base-uncased" # Англоязычная модель
PRETRAINED_MODEL_NAME = "bert-base-uncased"
DATA_FILE_PATH = "data/synthetic_nlu_data_en.jsonl" # Англоязычные данные
MODEL_SAVE_PATH = "models/slot_filler_bert_en" # Путь сохранения для англ. модели

# Параметры обучения
NUM_EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 128
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index # -100

class SlotFillingDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, label_map: Dict[str, int], max_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item_idx):
        record = self.records[item_idx]
        text_sentence = record["text"]
        # IOB-теги из файла соответствуют токенам, которые СГЕНЕРИРОВАЛ training_data_generator.py
        # с использованием того же токенизатора BERT.
        source_iob_tags_str = record["iob_tags"]
        # source_tokens = record["tokens"] # Токены из файла, для отладки или сверки

        encoding = self.tokenizer.encode_plus(
            text_sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # Метки должны быть выровнены с токенами, которые ПОЛУЧИЛИСЬ ЗДЕСЬ после encode_plus.
        # training_data_generator.py уже должен был создать source_iob_tags_str
        # с учетом [CLS] и [SEP] и правильной длины (или чуть больше/меньше MAX_SEQ_LENGTH,
        # но соответствующей своим source_tokens).
        labels_aligned = [PAD_TOKEN_LABEL_ID] * self.max_len

        # Проходим по токенам, полученным ЗДЕСЬ, и назначаем метки
        # Пропускаем [CLS] (первый токен) и [SEP] (последний не-PAD токен) для назначения меток слотов,
        # так как они не несут информации о слотах сущностей. Их метка будет PAD_TOKEN_LABEL_ID.

        # Находим индексы реальных токенов (не CLS, SEP, PAD)
        # `all_special_ids` включает и PAD, и CLS, и SEP, и UNK, и MASK

        # Итерация по токенам, которые мы получили при токенизации здесь
        # source_idx - это индекс в исходных метках из файла (source_iob_tags_str)
        # Он должен инкрементироваться только для токенов, которые были и в исходной разметке.
        # training_data_generator уже включает [CLS] и [SEP] в "tokens" и "iob_tags".

        for i in range(self.max_len):
            if attention_mask[i].item() == 0: # Это PAD токен
                labels_aligned[i] = PAD_TOKEN_LABEL_ID
                continue # Переходим к следующему токену

            # Если i-й токен не PAD, он может быть [CLS], [SEP] или обычным токеном.
            # source_iob_tags_str из файла уже содержит метки для [CLS] и [SEP] (обычно 'O').
            # Нам нужно просто скопировать соответствующую метку.
            if i < len(source_iob_tags_str):
                tag_str = source_iob_tags_str[i]
                labels_aligned[i] = self.label_map.get(tag_str, PAD_TOKEN_LABEL_ID) # Если метки нет, ставим PAD (не должно быть)
            else:
                # Это ситуация, когда MAX_SEQ_LENGTH здесь больше, чем длина токенов в файле
                # (что маловероятно, если MAX_SEQ_LENGTH в генераторе не был меньше).
                # Или если source_iob_tags_str был обрезан.
                # В любом случае, если мы вышли за пределы исходных меток, это PAD.
                labels_aligned[i] = PAD_TOKEN_LABEL_ID

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels_aligned, dtype=torch.long)
        }

def load_slot_data(data_path: str) -> List[Dict[str, Any]]:
    records = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line in {data_path}: {e} - Line: '{line.strip()}'")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}")
    return records

def get_all_iob_tags(records: List[Dict[str, Any]]) -> List[str]:
    unique_tags = set()
    for record in records:
        if "iob_tags" in record:
            for tag in record["iob_tags"]:
                unique_tags.add(tag)
    if "O" not in unique_tags: # 'O' (Outside) is fundamental
        unique_tags.add("O")
    return sorted(list(unique_tags))


def compute_slot_metrics(pred_logits_batch, true_labels_batch, label_map_inv):
    pred_ids_batch = np.argmax(pred_logits_batch, axis=2)

    true_predictions_str_all = [] # List of lists of string labels for predictions
    true_labels_str_all = []      # List of lists of string labels for true labels

    for i in range(true_labels_batch.shape[0]):
        pred_ids_sample = pred_ids_batch[i]
        label_ids_sample = true_labels_batch[i]

        active_preds_sample_str = []
        active_labels_sample_str = []
        for pred_id, label_id in zip(pred_ids_sample, label_ids_sample):
            if label_id != PAD_TOKEN_LABEL_ID:
                active_preds_sample_str.append(label_map_inv.get(pred_id, "O"))
                active_labels_sample_str.append(label_map_inv.get(label_id, "O"))

        if active_labels_sample_str:
            true_predictions_str_all.append(active_preds_sample_str)
            true_labels_str_all.append(active_labels_sample_str)

    if not true_labels_str_all:
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'report': "No non-PAD labels found in batch."}

    try:
        from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

        # Ensure all sublists have the same length for seqeval if that's an issue,
        # but seqeval handles lists of lists of strings correctly.
        precision = precision_score(true_labels_str_all, true_predictions_str_all, zero_division=0)
        recall = recall_score(true_labels_str_all, true_predictions_str_all, zero_division=0)
        f1 = f1_score(true_labels_str_all, true_predictions_str_all, zero_division=0)
        report_str = "Report N/A (potential issue)"
        try:
            report_str = classification_report(true_labels_str_all, true_predictions_str_all, zero_division=0)
        except Exception as e_report:
            report_str = f"Could not generate seqeval classification_report: {e_report}"
            # print(f"Data for report: True: {true_labels_str_all}, Pred: {true_predictions_str_all}")


        flat_true_preds_str = [p for sublist in true_predictions_str_all for p in sublist]
        flat_true_labels_str = [l for sublist in true_labels_str_all for l in sublist]

        acc = accuracy_score(flat_true_labels_str, flat_true_preds_str) if flat_true_labels_str else 0.0

        return {
            'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'report': report_str
        }
    except ImportError:
        print("`seqeval` library not found. Calculating only token-level accuracy.")
        flat_preds_ids_epoch = []
        flat_labels_ids_epoch = []
        for i in range(true_labels_batch.shape[0]):
            for pred_id, label_id in zip(pred_ids_batch[i], true_labels_batch[i]):
                if label_id != PAD_TOKEN_LABEL_ID:
                    flat_preds_ids_epoch.append(pred_id)
                    flat_labels_ids_epoch.append(label_id)

        acc = accuracy_score(flat_labels_ids_epoch, flat_preds_ids_epoch) if flat_labels_ids_epoch else 0.0
        return {
            'accuracy': acc, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
            'report': "seqeval not installed, only token accuracy available."
        }


def train_epoch_slots(model, data_loader, optimizer, device, scheduler, label_map_inv):
    model = model.train()
    total_loss = 0

    epoch_logits_list = []
    epoch_true_labels_list = []

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

        epoch_logits_list.append(logits.detach().cpu().numpy())
        epoch_true_labels_list.append(labels.detach().cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if batch_idx > 0 and batch_idx % (len(data_loader)//10 if len(data_loader) > 10 else 1) == 0:
             print(f"  Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)

    # Concatenate results from all batches in the epoch
    final_epoch_logits = np.concatenate(epoch_logits_list, axis=0)
    final_epoch_true_labels = np.concatenate(epoch_true_labels_list, axis=0)

    metrics = compute_slot_metrics(final_epoch_logits, final_epoch_true_labels, label_map_inv)
    return metrics['f1'], avg_loss, metrics


def eval_model_slots(model, data_loader, device, label_map_inv):
    model = model.eval()
    total_loss = 0

    epoch_logits_list = []
    epoch_true_labels_list = []

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
            epoch_logits_list.append(logits.detach().cpu().numpy())
            epoch_true_labels_list.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    final_epoch_logits = np.concatenate(epoch_logits_list, axis=0)
    final_epoch_true_labels = np.concatenate(epoch_true_labels_list, axis=0)
    metrics = compute_slot_metrics(final_epoch_logits, final_epoch_true_labels, label_map_inv)
    return metrics['f1'], avg_loss, metrics


def main():
    print("Starting Slot Filler Model Training (English)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records = load_slot_data(DATA_FILE_PATH)
    if not records: return

    unique_iob_tags = get_all_iob_tags(records)
    if not unique_iob_tags or len(unique_iob_tags) <= 1:
        print(f"Not enough unique IOB tags found ({len(unique_iob_tags)}). Check data generation. Tags: {unique_iob_tags}")
        return

    slot_label_to_id = {tag: i for i, tag in enumerate(unique_iob_tags)}
    id_to_slot_label = {i: tag for tag, i in slot_label_to_id.items()}
    num_slot_classes = len(unique_iob_tags)

    print(f"Number of unique IOB slot tags: {num_slot_classes}")
    # print(f"Slot Label to ID mapping (first 5): {list(slot_label_to_id.items())[:5]}")

    train_records, val_records = train_test_split(records, test_size=0.15, random_state=42)
    print(f"Training samples: {len(train_records)}, Validation samples: {len(val_records)}")
    if not train_records or not val_records:
        print("Error: Not enough data for training or validation split. Need more generated samples.")
        return

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    train_dataset = SlotFillingDataset(train_records, tokenizer, slot_label_to_id, MAX_SEQ_LENGTH)
    val_dataset = SlotFillingDataset(val_records, tokenizer, slot_label_to_id, MAX_SEQ_LENGTH)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    model = AutoModelForTokenClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=num_slot_classes,
        id2label=id_to_slot_label,
        label2id=slot_label_to_id
    )
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * NUM_EPOCHS
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps
    )

    print("\n--- Starting Training ---")
    best_val_f1 = -1.0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_f1, train_loss, train_metrics = train_epoch_slots(
            model, train_data_loader, optimizer, device, scheduler, id_to_slot_label
        )
        print(f"Train loss: {train_loss:.4f}, Train F1 (entity): {train_f1:.4f}")
        # print(f"Full Train metrics: {train_metrics.get('report', 'N/A')}")


        val_f1, val_loss, val_metrics = eval_model_slots(
            model, val_data_loader, device, id_to_slot_label
        )
        print(f"Val loss: {val_loss:.4f}, Val F1 (entity): {val_f1:.4f}")
        val_report = val_metrics.get('report', 'N/A')
        # print(f"Full Validation metrics report:\n{val_report}") # Can be very long

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"New best validation F1: {best_val_f1:.4f}. Saving model...")
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            with open(os.path.join(MODEL_SAVE_PATH, "slot_label_map.json"), 'w', encoding='utf-8') as f:
                json.dump({"slot_label_to_id": slot_label_to_id, "id_to_slot_label": id_to_slot_label},
                          f, ensure_ascii=False, indent=4)

    print("\n--- Training Finished ---")
    print(f"Best validation F1 score: {best_val_f1:.4f}")
    print(f"Model, tokenizer and label map saved to {MODEL_SAVE_PATH} (if F1 improved or first save attempt)")

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE_PATH):
        print(f"ERROR: Data file not found at {DATA_FILE_PATH}")
        print(f"Please run src/training_data_generator.py first to generate '{DATA_FILE_PATH}'.")
    else:
        try:
            import seqeval
        except ImportError:
            print("WARNING: `seqeval` library not found. Slot filling metrics (F1, P, R) will be basic or zero.")
            print("For proper evaluation, please install it: pip install seqeval")
        main()
