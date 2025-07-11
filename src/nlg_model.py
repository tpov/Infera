import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Any, Optional, List
import os # Добавлен os для проверки пути

# --- Конфигурация для NLG модели (английский язык) ---
NLG_TOKENIZER_NAME = "facebook/bart-base"
NLG_PRETRAINED_MODEL_NAME = "facebook/bart-base"
NLG_MODEL_SAVE_PATH = "models/nlg_bart_en"

THINKING_OUTPUT_SEPARATOR = "%%IDEAS:%%"

class NLGModel:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NLGModel using device: {self.device}")

        self.tokenizer = None
        self.model = None

        self._load_model() # Пытаемся загрузить модель при инициализации
        if self.model is None or self.tokenizer is None:
            print("NLGModel initialized WITHOUT a loaded model. Generation will use placeholders.")
        else:
            print(f"NLGModel initialized and model '{self.model.name_or_path}' loaded successfully.")

    def _load_model(self, model_path: str = NLG_MODEL_SAVE_PATH):
        load_path = model_path
        # Проверяем, существует ли путь и является ли он директорией (Hugging Face сохраняет модели в директории)
        if not os.path.exists(load_path) or not os.path.isdir(load_path):
            print(f"Warning: Fine-tuned NLG model not found at directory {load_path}. Attempting to load base '{NLG_PRETRAINED_MODEL_NAME}'.")
            load_path = NLG_PRETRAINED_MODEL_NAME
        try:
            print(f"Loading NLG tokenizer from: {load_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            print(f"Loading NLG model from: {load_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path).to(self.device)
            self.model.eval()
            print(f"NLG model '{load_path}' loaded successfully.")
        except Exception as e:
            print(f"FATAL: Could not load NLG model or tokenizer from '{load_path}'. Error: {e}")
            self.tokenizer = None
            self.model = None
            # Не будем перевыбрасывать исключение здесь, чтобы остальная часть приложения могла
            # попытаться запуститься и выдать ошибку на более высоком уровне или работать с заглушками.
            # Но в консоли будет видно FATAL.

    def _prepare_input_for_nlg(self, nlu_result: Dict[str, Any], state: Dict[str, Any], task_type: str = "GENERATE_RESPONSE") -> str:
        intent_str = nlu_result.get("intent", "none")
        slots_parts = []
        for slot in nlu_result.get("slots", []):
            slots_parts.append(f"{slot.get('type','unknown_type')}:{slot.get('value','unknown_value')}")
        slots_str = ", ".join(slots_parts) if slots_parts else "none"

        entities_parts = []
        for entity_name, attrs in state.get("entities", {}).items():
            attr_strs = []
            for attr_name, attr_val in attrs.items():
                attr_strs.append(f"{attr_name}:{attr_val}")
            entities_parts.append(f"{entity_name}({', '.join(attr_strs)})")
        entities_str = ", ".join(entities_parts) if entities_parts else "none"

        goals_parts = []
        for goal in state.get("active_goals", []):
            goal_desc_parts = [f"{k}:{str(v)}" for k,v in goal.items()]
            goals_parts.append(f"goal({', '.join(goal_desc_parts)})")
        goals_parts = []
        for goal in state.get("active_goals", []):
            goal_desc_parts = [f"{k}:{str(v)}" for k,v in goal.items()]
            goals_parts.append(f"goal({', '.join(goal_desc_parts)})")
        goals_str = ", ".join(goals_parts) if goals_parts else "none"

        facts_parts = []
        for fact in state.get("facts", []):
            if isinstance(fact, dict):
                fact_detail_parts = [f"{k}:{str(v)}" for k,v in fact.items()]
                facts_parts.append(f"fact({', '.join(fact_detail_parts)})")
            elif isinstance(fact, str):
                facts_parts.append(f"fact({fact})")
        facts_str = ", ".join(facts_parts) if facts_parts else "none"

        # Добавляем meta информацию, если она есть в nlu_result или передана через kwargs
        # (kwargs напрямую в _prepare_input_for_nlg не передаются, но их можно извлечь из nlu_result, если LogicalController их туда кладет)
        meta_info_parts = []
        if nlu_result.get("meta_missing_slot"):
            meta_info_parts.append(f"missing_focus:{nlu_result['meta_missing_slot']}")
        if nlu_result.get("meta_prompt_info"): # Если LogicalController добавил это в nlu_result
             meta_info_parts.append(f"prompt_hint:{nlu_result['meta_prompt_info']}")

        meta_info_str = " | ".join(meta_info_parts) if meta_info_parts else "none"

        input_text = f"task: {task_type} | intent: {intent_str} | slots: {slots_str} | entities: {entities_str} | facts: {facts_str} | goals: {goals_str} | meta: {meta_info_str}"
        # print(f"DEBUG NLG Input: {input_text}")
        return input_text

    def generate_text(self, nlu_result: Dict[str, Any], current_state: Dict[str, Any], task_type: str = "GENERATE_RESPONSE", **generation_kwargs) -> str:
        # Извлекаем meta_prompt_info из generation_kwargs и добавляем в nlu_result, если его там нет,
        # чтобы _prepare_input_for_nlg мог его использовать.
        # Это немного хак, лучше бы _prepare_input_for_nlg принимал **kwargs.
        # Но для простоты пока так.
        if "meta_prompt_info" in generation_kwargs and "meta_prompt_info" not in nlu_result:
            nlu_result_augmented = nlu_result.copy() # Не меняем исходный nlu_result
            nlu_result_augmented["meta_prompt_info"] = generation_kwargs["meta_prompt_info"]
        else:
            nlu_result_augmented = nlu_result

        if not self.model or not self.tokenizer:
            print("NLGModel: Model not loaded. Returning placeholder response.")
            if task_type == "REQUEST_CLARIFICATION":
                missing_slot = nlu_result_augmented.get("meta_missing_slot", "required information")
                return f"I seem to be missing some information. Could you please provide the {missing_slot}?"
            elif task_type == "THINK_AND_PLAN":
                return f"Let me consider that... {THINKING_OUTPUT_SEPARATOR} Suggestion: explore alternative approaches. Question: what are the constraints?"
            return "I'm currently unable to generate a full response. (NLG Model Not Loaded)"

        input_text = self._prepare_input_for_nlg(nlu_result_augmented, current_state, task_type) # Используем дополненный nlu_result

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="max_length"
        ).to(self.device)

        default_generation_kwargs = {
            "max_length": 150,
            "min_length": 5,
            "num_beams": 4,
            "early_stopping": True,
        }
        final_generation_kwargs = {**default_generation_kwargs, **generation_kwargs}

        try:
            output_ids = self.model.generate(inputs['input_ids'], **final_generation_kwargs)
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception as e:
            print(f"Error during NLG generation: {e}")
            generated_text = "Sorry, I had trouble generating a response."

        return generated_text.strip()

if __name__ == '__main__':
    print("Initializing NLGModel...")

    nlg_system = NLGModel() # _load_model() вызовется внутри __init__

    if nlg_system.model is None or nlg_system.tokenizer is None:
        print("\nNLG Model or Tokenizer not loaded. Test will use placeholder responses.")
    else:
        print(f"\nNLG Model '{nlg_system.model.name_or_path}' seems to be loaded for test.")

    sample_nlu_result_1 = {"intent": "QUERY_COUNT_RESULT", "slots": [{"type": "ENTITY_NAME", "value": "apples"}, {"type": "COUNT", "value": "5"}]}
    sample_state_1 = {"entities": {"apples": {"count": 5, "color": "red", "location": "table"}}, "active_goals": []}

    sample_nlu_result_2 = {"intent": "ADD_ENTITY", "slots": [{"type": "ENTITY_NAME", "value": "books"}], "meta_missing_slot": "location"}
    sample_state_2 = {"entities": {"apples": {"count": 5, "color": "red", "location": "table"}}, "active_goals": [{"goal_id": "clarify123", "goal_type": "CLARIFY_LOCATION", "for_intent": "ADD_ENTITY", "entity_name": "books", "status": "pending_user_clarification"}]}

    sample_nlu_result_3 = {"intent": "CREATE_BUSINESS_PLAN", "slots": [{"type": "FIELD", "value": "online store"}]}
    sample_state_3 = {"entities": {},
                      "active_goals": [{"goal_id": "bp1", "goal_type": "PLAN_BUSINESS", "missing_info": ["budget", "product_type"], "status": "pending_system_thought"}]}

    print("\n--- Testing NLG Text Generation ---")

    response1 = nlg_system.generate_text(sample_nlu_result_1, sample_state_1, task_type="GENERATE_RESPONSE")
    print(f"\nInput for Response 1 (State + NLU for count result):")
    print(f"  NLU: {sample_nlu_result_1}")
    print(f"  State Entities: {sample_state_1['entities']}")
    print(f"Generated Response 1: {response1}")

    response2 = nlg_system.generate_text(sample_nlu_result_2, sample_state_2, task_type="REQUEST_CLARIFICATION")
    print(f"\nInput for Response 2 (State + NLU for clarification request):")
    print(f"  NLU: {sample_nlu_result_2}") # meta_missing_slot используется заглушкой
    print(f"  State Goals: {sample_state_2['active_goals']}")
    print(f"Generated Response 2 (Clarification): {response2}")

    response3 = nlg_system.generate_text(sample_nlu_result_3, sample_state_3, task_type="THINK_AND_PLAN")
    print(f"\nInput for Response 3 (State + NLU for thinking output):")
    print(f"  NLU: {sample_nlu_result_3}")
    print(f"  State Goals: {sample_state_3['active_goals']}")
    print(f"Generated Response 3 (Thinking Output): {response3}")
    if THINKING_OUTPUT_SEPARATOR in response3: # Используем константу класса
        parts = response3.split(THINKING_OUTPUT_SEPARATOR, 1)
        print(f"  Extracted Question Part: {parts[0].strip()}")
        print(f"  Extracted Ideas Part: {parts[1].strip() if len(parts) > 1 else 'No ideas part found'}")

    print("\nNLGModel script finished.")
