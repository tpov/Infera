import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import numpy as np
from typing import List, Dict, Any, Optional # Добавлены Optional, List, Dict, Any

# Пути к сохраненным моделям (должны соответствовать путям в скриптах обучения)
INTENT_MODEL_PATH = "models/intent_classifier_bert_en"
SLOT_MODEL_PATH = "models/slot_filler_bert_en"
TOKENIZER_NAME = "bert-base-uncased"

# StateManager больше не импортируется и не используется в NLUController
# try:
#     from state_manager import StateManager
# except ImportError:
#     # ... (старый код импорта для __main__)

class NLUController:
    """
    Контроллер для NLU: понимание намерений и извлечение слотов/сущностей из текста.
    Возвращает структурированный JSON, соответствующий `target_structured_output` из генератора данных.
    """
    def __init__(self, state_manager: Optional[Any] = None, device=None): # state_manager теперь опционален и не используется
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NLUController using device: {self.device}")

        self.tokenizer = None
        self.intent_model = None
        self.slot_model = None
        self.intent_id_to_label = None # type: Optional[Dict[str, str]]
        self.slot_id_to_label = None   # type: Optional[Dict[str, str]]

        self._load_models()
        if self.tokenizer and self.intent_model and self.slot_model:
            print("NLUController initialized successfully with models.")
        else:
            print("NLUController initialization FAILED: one or more models/tokenizer could not be loaded.")


    def _load_models(self):
        """Загружает обученные модели и токенизатор."""
        try:
            # Токенизатор может быть сохранен с любой моделью, грузим из одного места для консистентности
            # Предпочтительнее грузить из SLOT_MODEL_PATH, если он есть, т.к. он мог быть адаптирован для токенов
            if os.path.exists(SLOT_MODEL_PATH) and os.path.isdir(SLOT_MODEL_PATH):
                print(f"Loading tokenizer from: {SLOT_MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(SLOT_MODEL_PATH)
            elif os.path.exists(INTENT_MODEL_PATH) and os.path.isdir(INTENT_MODEL_PATH):
                print(f"Loading tokenizer from: {INTENT_MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
            else:
                print(f"Attempting to load tokenizer by name: {TOKENIZER_NAME}")
                self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

            if os.path.exists(INTENT_MODEL_PATH) and os.path.isdir(INTENT_MODEL_PATH):
                print(f"Loading intent model from: {INTENT_MODEL_PATH}")
                self.intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH).to(self.device)
                self.intent_model.eval()
                intent_map_path = os.path.join(INTENT_MODEL_PATH, "intent_label_map.json")
                if os.path.exists(intent_map_path):
                    with open(intent_map_path, 'r', encoding='utf-8') as f:
                        self.intent_id_to_label = json.load(f).get("id_to_intent") # Убедимся, что берем правильную карту
                else: print(f"Warning: Intent label map not found at {intent_map_path}")
            else: print(f"Warning: Intent model not found at {INTENT_MODEL_PATH}.")

            if os.path.exists(SLOT_MODEL_PATH) and os.path.isdir(SLOT_MODEL_PATH):
                print(f"Loading slot filler model from: {SLOT_MODEL_PATH}")
                self.slot_model = AutoModelForTokenClassification.from_pretrained(SLOT_MODEL_PATH).to(self.device)
                self.slot_model.eval()
                slot_map_path = os.path.join(SLOT_MODEL_PATH, "slot_label_map.json")
                if os.path.exists(slot_map_path):
                    with open(slot_map_path, 'r', encoding='utf-8') as f:
                        self.slot_id_to_label = json.load(f).get("id_to_label") # Убедимся, что берем правильную карту
                else: print(f"Warning: Slot label map not found at {slot_map_path}")
            else: print(f"Warning: Slot model not found at {SLOT_MODEL_PATH}.")

        except Exception as e:
            print(f"Error loading NLU models/tokenizer: {e}")
            # Устанавливаем в None, чтобы проверки в predict_nlu сработали
            self.tokenizer = None
            self.intent_model = None
            self.slot_model = None


    def _decode_iob_tags(self, tokens: List[str], iob_ids: List[int],
                         attention_mask: List[int], token_offsets: List[Tuple[int,int]],
                         original_text: str) -> List[Dict[str, Any]]:
        if not self.slot_id_to_label or not self.tokenizer: return []

        slots = []
        current_slot_tokens_indices = []
        current_slot_type = None

        for i in range(len(iob_ids)):
            if attention_mask[i] == 0: continue # PAD
            # Игнорируем CLS/SEP для формирования значений слотов, но не для логики B-/I-
            # is_special_token = tokens[i] in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]

            label_id_str = str(iob_ids[i])
            label_str = self.slot_id_to_label.get(label_id_str, "O")

            if label_str.startswith("B-"):
                if current_slot_type and current_slot_tokens_indices: # Закрываем предыдущий слот
                    start_char = token_offsets[current_slot_tokens_indices[0]][0]
                    end_char = token_offsets[current_slot_tokens_indices[-1]][1]
                    slot_value = original_text[start_char:end_char]
                    slots.append({"type": current_slot_type, "value": slot_value.strip(), "char_spans": (start_char, end_char)})

                current_slot_tokens_indices = [i]
                current_slot_type = label_str[2:]
            elif label_str.startswith("I-"):
                if current_slot_type == label_str[2:] and current_slot_tokens_indices: # Продолжение текущего
                    current_slot_tokens_indices.append(i)
                else: # I-метка другого типа или без B- -> ошибка, закрываем старый, этот O
                    if current_slot_type and current_slot_tokens_indices:
                        start_char = token_offsets[current_slot_tokens_indices[0]][0]
                        end_char = token_offsets[current_slot_tokens_indices[-1]][1]
                        slot_value = original_text[start_char:end_char]
                        slots.append({"type": current_slot_type, "value": slot_value.strip(), "char_spans": (start_char, end_char)})
                    current_slot_tokens_indices = []
                    current_slot_type = None
            else: # "O"
                if current_slot_type and current_slot_tokens_indices: # Закрываем предыдущий слот
                    start_char = token_offsets[current_slot_tokens_indices[0]][0]
                    end_char = token_offsets[current_slot_tokens_indices[-1]][1]
                    slot_value = original_text[start_char:end_char]
                    slots.append({"type": current_slot_type, "value": slot_value.strip(), "char_spans": (start_char, end_char)})
                current_slot_tokens_indices = []
                current_slot_type = None

        if current_slot_type and current_slot_tokens_indices: # Закрываем последний открытый слот
            start_char = token_offsets[current_slot_tokens_indices[0]][0]
            end_char = token_offsets[current_slot_tokens_indices[-1]][1]
            slot_value = original_text[start_char:end_char]
            slots.append({"type": current_slot_type, "value": slot_value.strip(), "char_spans": (start_char, end_char)})
        return slots

    def _build_structured_output(self, intent_label: str, extracted_slots: List[Dict[str, Any]], source_text: str) -> Dict[str, Any]:
        """
        Собирает сложный структурированный JSON из предсказанного намерения и извлеченных слотов.
        Эта логика должна быть зеркальным отражением `target_json_builder` из `training_data_generator.py`.
        """
        output = {"overall_intent": intent_label, "source_text": source_text}
        # Вспомогательная функция для поиска слотов
        def get_slot_val(slot_type: str, default: Any = None) -> Any:
            return next((s["value"] for s in extracted_slots if s["type"] == slot_type), default)
        def get_all_slots(slot_type: str) -> List[Dict[str,Any]]:
            return [s for s in extracted_slots if s["type"] == slot_type]

        # TODO: Реализовать логику сборки для КАЖДОГО интента, который предполагает сложную структуру.
        # Это будет большая работа, требующая точного соответствия target_json_builder.
        # Ниже приведены очень УПРОЩЕННЫЕ примеры, которые нужно будет значительно доработать.

        if intent_label == "ADD_ENTITY_WITH_DETAILS":
            output["action_details"] = {"verb": "add"}
            output["entities_involved"] = [{
                "name": get_slot_val("ENTITY_NAME_PLURAL"), # Генератор использует ENTITY_NAME_PLURAL
                "attributes": {
                    "count": get_slot_val("COUNT"),
                    "location": get_slot_val("LOCATION"),
                    "color": get_slot_val("ATTRIBUTE_VALUE")
                }
            }]
        elif intent_label == "QUERY_COUNT":
            output["query_type"] = "GET_ATTRIBUTE"
            output["entity_name"] = get_slot_val("ENTITY_NAME_PLURAL")
            output["attribute_to_query"] = "count"
            loc = get_slot_val("LOCATION")
            if loc: output["conditions"] = [{"location": loc}]

        elif intent_label == "PROCESS_LOGICAL_STATEMENT_IMPLICATION":
            output["statement_type"] = "CONDITIONAL_RULE"
            output["condition"] = {"raw_text": get_slot_val("CONDITION_PHRASE")}
            output["consequence"] = {"raw_text": get_slot_val("CONSEQUENCE_PHRASE")}
            # Если бы NLU извлекал еще и ASSERTED_CONDITION_PHRASE для факта:
            # asserted = get_slot_val("ASSERTED_CONDITION_PHRASE")
            # if asserted: output["asserted_fact_text"] = asserted

        elif intent_label == "SIMULATE_SCENARIO_CHANGE":
            # Эта логика должна быть намного сложнее, чтобы правильно сгруппировать
            # initial_entities и entities_to_add из плоского списка extracted_slots.
            # Например, нужно будет смотреть на порядок слотов, их типы (ENTITY_A, ENTITY_B, ENTITY_C),
            # чтобы правильно сопоставить их с initial_entities и entities_to_add.
            # Пока что это очень грубая заглушка.
            output["scenario_context"] = {
                "initial_entities": [{"name": get_slot_val("ENTITY_C_PLURAL"), "count": get_slot_val("COUNT_C")}],
                "actions": [{"verb": "add", "entities_to_add": [
                    {"name": get_slot_val("ENTITY_A_PLURAL"), "count": get_slot_val("COUNT_A")},
                    {"name": get_slot_val("ENTITY_B_SINGULAR") or get_slot_val("ENTITY_B_PLURAL"),
                     "count": get_slot_val("COUNT_B", "1")} # По умолчанию 1, если не извлечен
                ]}]
            }
            output["query_type"] = "PREDICT_OUTCOME"
        else:
            # Для других интентов, если нет специальной структуры, можно просто вернуть "сырые" слоты
            output["extracted_slots_flat"] = extracted_slots
            # Но лучше иметь явную структуру для каждого интента, как в target_json_builder

        # print(f"DEBUG _build_structured_output: {json.dumps(output, indent=2)}")
        return output

    def predict_nlu(self, text: str) -> Dict[str, Any]:
        """ Выполняет NLU: определяет намерение и извлекает слоты, затем строит сложный JSON. """
        if not self.tokenizer or not self.intent_model or not self.slot_model or \
           not self.intent_id_to_label or not self.slot_id_to_label:
            return {"overall_intent": "NLU_ERROR", "source_text": text, "error": "NLU models, tokenizer, or label maps not loaded."}

        inputs = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=128, # TODO: Использовать константу MAX_SEQ_LENGTH
            padding='max_length', truncation=True, return_attention_mask=True,
            return_tensors='pt', return_offsets_mapping=True # Нужны offset_mapping для _decode_iob_tags
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        offset_mapping = inputs['offset_mapping'].squeeze().tolist() # Для использования в _decode_iob_tags

        with torch.no_grad():
            intent_logits = self.intent_model(input_ids, attention_mask=attention_mask).logits
            slot_logits = self.slot_model(input_ids, attention_mask=attention_mask).logits

        predicted_intent_id = str(torch.argmax(intent_logits, dim=1).item()) # Ключи в карте - строки
        predicted_intent_label = self.intent_id_to_label.get(predicted_intent_id, "UNKNOWN_INTENT")

        predicted_slot_ids = torch.argmax(slot_logits, dim=2)[0].cpu().tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        attention_mask_list = attention_mask[0].cpu().tolist()

        # Используем original_text для извлечения значений слотов по char_spans
        extracted_slots_with_spans = self._decode_iob_tags(tokens, predicted_slot_ids, attention_mask_list, offset_mapping, text)

        # Собираем сложный структурированный вывод
        final_structured_output = self._build_structured_output(predicted_intent_label, extracted_slots_with_spans, text)

        return final_structured_output

    # process_sentence удален, так как NLUController отвечает только за NLU, а не за обработку состояния.
    # Логический контроллер будет вызывать predict_nlu().

if __name__ == '__main__':
    print("Initializing components for NLUController test...")
    if not os.path.exists(INTENT_MODEL_PATH) or not os.path.exists(SLOT_MODEL_PATH):
        print(f"ERROR: Trained models not found! Ensure '{INTENT_MODEL_PATH}' and '{SLOT_MODEL_PATH}' exist.")
    else:
        nlu_controller = NLUController()
        if nlu_controller.tokenizer and nlu_controller.intent_model and nlu_controller.slot_model:
            print("\n--- Testing NLUController.predict_nlu() (English) ---")
            test_sentences = [
                "hello",
                "in the kitchen there are 5 red apples",
                "how many apples",
                "what happens if you add 3 more elephants and a tiger to two elephants",
                "if it rains then the ground is wet",
                "it is raining",
                "goodbye"
            ]
            for sentence in test_sentences:
                structured_nlu_output = nlu_controller.predict_nlu(sentence)
                print(f"\n  User: {sentence}")
                print(f"  NLU Output (Structured JSON):")
                print(json.dumps(structured_nlu_output, indent=2, ensure_ascii=False))
                print("-" * 30)
        else:
            print("NLUController could not be fully initialized.")
    print("\nNLUController test finished.")
