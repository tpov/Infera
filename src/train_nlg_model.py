import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_scheduler, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import numpy as np
# Для метрик BLEU/ROUGE может понадобиться `nltk` или `rouge_score`, `evaluate`
# pip install nltk rouge_score evaluate sacrebleu
import evaluate # Hugging Face Evaluate library

# --- Конфигурация ---
# Используем те же имена, что и в nlg_model.py для базовой модели
NLG_TOKENIZER_NAME = "facebook/bart-base"
NLG_PRETRAINED_MODEL_NAME = "facebook/bart-base"
NLG_DATA_FILE_PATH = "data/synthetic_nlg_bart_data_en.jsonl"
NLG_MODEL_SAVE_PATH = "models/nlg_bart_en" # Куда сохраняем fine-tuned модель

# Параметры обучения
NUM_EPOCHS = 3 # Fine-tuning обычно требует немного эпох
BATCH_SIZE = 8  # Seq2Seq модели более требовательны к памяти, начнем с меньшего батча
LEARNING_RATE = 5e-5
MAX_INPUT_LENGTH = 512  # Максимальная длина входной последовательности для энкодера
MAX_TARGET_LENGTH = 128 # Максимальная длина генерируемой последовательности (ответа)

# Глобальная инициализация токенизатора и метрик
tokenizer = AutoTokenizer.from_pretrained(NLG_TOKENIZER_NAME)
rouge_metric = evaluate.load("rouge")
# bleu_metric = evaluate.load("sacrebleu") # SacreBLEU для BLEU

class NLGDataset(Dataset):
    """Кастомный датасет для обучения Seq2Seq NLG модели."""
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_input_len: int, max_target_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        record = self.data[item_idx]
        input_text = record["input_text"]
        target_text = record["target_text"]

        # Токенизация входа для энкодера
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length", # Паддинг до max_input_len
            truncation=True,
            return_tensors="pt"
        )

        # Токенизация выхода (целевого текста) для декодера
        # Метки для декодера должны быть подготовлены так, чтобы модель училась их генерировать.
        # Токенизатор BART (и T5) автоматически обрабатывает сдвиг меток для декодера,
        # если labels передаются в model.forward().
        with self.tokenizer.as_target_tokenizer(): # Важно для некоторых токенизаторов Seq2Seq
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_len,
                padding="max_length", # Паддинг до max_target_len
                truncation=True,
                return_tensors="pt"
            ).input_ids # Нам нужны только input_ids для labels

        # Заменяем PAD токены в labels на -100, чтобы они игнорировались в функции потерь
        labels[labels == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels.flatten() # (seq_len)
        model_inputs["input_ids"] = model_inputs["input_ids"].flatten() # (seq_len)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].flatten() # (seq_len)

        return model_inputs


def load_nlg_data(data_path: str) -> List[Dict[str, str]]:
    """Загружает пары (input_text, target_text) из JSONL файла."""
    data_pairs = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "input_text" in record and "target_text" in record:
                        data_pairs.append({"input_text": record["input_text"], "target_text": record["target_text"]})
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line in {data_path}: {e} - Line: '{line.strip()}'")
    except FileNotFoundError:
        print(f"ERROR: NLG Data file not found at {data_path}")
    return data_pairs

def compute_nlg_metrics(eval_preds):
    """Вычисляет метрики ROUGE (и BLEU) для оценки генерации."""
    preds, labels = eval_preds # preds - это logits или уже сгенерированные ID

    # Декодируем предсказанные ID и истинные ID в текст
    # PAD токены нужно заменить на что-то или пропустить при декодировании
    # labels могут содержать -100, их нужно заменить на pad_token_id перед декодированием
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # ROUGE ожидает список строк для предсказаний и список списков строк для референсов (если их несколько на пример)
    # У нас один референс на пример.
    decoded_labels_rouge = [label.strip() for label in decoded_labels]
    decoded_preds_rouge = [pred.strip() for pred in decoded_preds]

    rouge_results = rouge_metric.compute(predictions=decoded_preds_rouge, references=decoded_labels_rouge, use_stemmer=True)

    # BLEU (SacreBLEU)
    # decoded_labels_bleu = [[label.strip()] for label in decoded_labels] # SacreBLEU ожидает список списков референсов
    # decoded_preds_bleu = [pred.strip() for pred in decoded_preds]
    # bleu_results = bleu_metric.compute(predictions=decoded_preds_bleu, references=decoded_labels_bleu)

    # Собираем метрики
    result = {key: value for key, value in rouge_results.items()}
    # result["bleu"] = bleu_results["score"]

    # Длина предсказаний (для информации)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}


def main():
    print("Starting NLG Model (BART) Fine-tuning (English)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Загрузка данных
    data_pairs = load_nlg_data(NLG_DATA_FILE_PATH)
    if not data_pairs:
        print(f"No data loaded from {NLG_DATA_FILE_PATH}. Exiting.")
        return
    print(f"Loaded {len(data_pairs)} data pairs for NLG training.")

    # 2. Разделение данных
    train_data, val_data = train_test_split(data_pairs, test_size=0.1, random_state=42)
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    if not train_data or not val_data:
        print("Error: Not enough data for training or validation split.")
        return

    # 3. Создание датасетов
    # Токенизатор уже инициализирован глобально
    train_dataset = NLGDataset(train_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    val_dataset = NLGDataset(val_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    # 4. Инициализация модели
    model = AutoModelForSeq2SeqLM.from_pretrained(NLG_PRETRAINED_MODEL_NAME)
    model = model.to(device)

    # 5. Data Collator - для динамического паддинга в батче (особенно для Seq2Seq)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 6. Настройка обучения с использованием Hugging Face Trainer API
    # Это значительно упрощает цикл обучения, оценку, сохранение.
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

    training_args = Seq2SeqTrainingArguments(
        output_dir=NLG_MODEL_SAVE_PATH + "_trainer_checkpoints", # Директория для чекпоинтов
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=max(1, int( (len(train_dataset) // BATCH_SIZE) * NUM_EPOCHS * 0.1)), # 10% warmup
        weight_decay=0.01,
        logging_dir='./logs_nlg',
        logging_steps=max(1, (len(train_dataset) // BATCH_SIZE) // 10), # Логировать ~10 раз за эпоху
        evaluation_strategy="epoch", # Оценка в конце каждой эпохи
        save_strategy="epoch",       # Сохранение в конце каждой эпохи
        load_best_model_at_end=True, # Загрузить лучшую модель по метрике в конце
        metric_for_best_model="rougeL", # Используем ROUGE-L для выбора лучшей модели
        predict_with_generate=True,  # Важно для Seq2Seq моделей для генерации текста при оценке
        fp16=torch.cuda.is_available(), # Использовать fp16, если доступна CUDA
        # optim="adamw_torch", # Используем AdamW из PyTorch
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_nlg_metrics
    )

    print("\n--- Starting Fine-tuning using Trainer ---")
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("Training interrupted.")
        return

    print("\n--- Fine-tuning Finished ---")

    # Оценка лучшей модели
    eval_results = trainer.evaluate()
    print(f"Evaluation results for the best model: {eval_results}")

    # Сохранение лучшей модели и токенизатора
    if not os.path.exists(NLG_MODEL_SAVE_PATH):
        os.makedirs(NLG_MODEL_SAVE_PATH, exist_ok=True)

    trainer.save_model(NLG_MODEL_SAVE_PATH) # Сохраняет модель и токенизатор
    # tokenizer.save_pretrained(NLG_MODEL_SAVE_PATH) # Trainer должен это делать сам

    print(f"Fine-tuned NLG model and tokenizer saved to {NLG_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    if not os.path.exists(NLG_DATA_FILE_PATH):
        print(f"ERROR: NLG Training Data file not found at {NLG_DATA_FILE_PATH}")
        print(f"Please run src/nlg_training_data_generator.py first to generate '{NLG_DATA_FILE_PATH}'.")
    else:
        # Проверка наличия необходимых библиотек для метрик
        try:
            import nltk
            # nltk.download('punkt', quiet=True) # Для ROUGE нужен токенизатор punkt
        except ImportError:
            print("WARNING: `nltk` library not found. ROUGE metric might not work correctly.")
            print("Please install it: pip install nltk")
        try:
            import rouge_score
        except ImportError:
             print("WARNING: `rouge_score` library not found. ROUGE metric will not work.")
             print("Please install it: pip install rouge_score")
        try:
            import sacrebleu
        except ImportError:
            print("INFO: `sacrebleu` library not found. BLEU metric will not be calculated.")
            # print("You can install it via: pip install sacrebleu")

        main()
