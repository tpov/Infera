import json
import random
import re
import os
from typing import List, Dict, Any, Tuple, Optional

# Англоязычная модель BERT
DEFAULT_TOKENIZER_NAME = "bert-base-uncased"

class TrainingDataGenerator:
    """
    Генерирует синтетические данные для обучения NLU моделей (классификация намерений и извлечение слотов).
    Данные включают:
    - text: Сгенерированное предложение.
    - intent: Метка намерения.
    - tokens: Список токенов (от BERT-совместимого токенизатора).
    - iob_tags: Список IOB-меток для каждого токена.
    - target_structured_output: Целевой JSON, который NLUController должен формировать
                                 ПОСЛЕ предсказаний BERT-моделей (интент + IOB-теги)
                                 и пост-обработки. Это для эталонной оценки всего NLU пайплайна.
    """
    def __init__(self, bert_tokenizer_name: str = DEFAULT_TOKENIZER_NAME):
        self.intents_config = self._get_intents_config()
        self.slot_fillers = self._get_slot_fillers()
        self.tokenizer_name = bert_tokenizer_name

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            print(f"Tokenizer '{self.tokenizer_name}' loaded successfully.")
        except ImportError:
            print("CRITICAL ERROR: `transformers` library not found. This script requires `transformers`.")
            print("Please install it: pip install transformers torch")
            self.tokenizer = None
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load tokenizer '{self.tokenizer_name}': {e}.")
            self.tokenizer = None

    def _get_intents_config(self) -> List[Dict[str, Any]]:
        """
        Определяет конфигурацию намерений: шаблоны, связанные слоты и пример того,
        как может выглядеть целевой структурированный вывод для этого интента.
        Поле 'target_json_example_structure_builder' - это функция, которая демонстрирует
        построение сложного JSON. В реальном генераторе она будет использоваться для создания
        'target_structured_output' для каждого сгенерированного примера.
        """
        config = [
            {
                "intent": "ADD_ENTITY_WITH_DETAILS",
                "patterns": [
                    "in the <LOCATION> there are <COUNT> <ATTRIBUTE_VALUE> <ENTITY_NAME_PLURAL>",
                    "add <COUNT> <ENTITY_NAME_PLURAL> with color <ATTRIBUTE_VALUE> to the <LOCATION>",
                    "put <COUNT> <ATTRIBUTE_VALUE> <ENTITY_NAME_PLURAL> on the <LOCATION>",
                ],
                "slots": ["LOCATION", "COUNT", "ATTRIBUTE_VALUE", "ENTITY_NAME_PLURAL"],
                "target_json_builder": lambda slots: {
                    "action_type": "ADD_UPDATE_ENTITY",
                    "entities": [{
                        "name": slots.get("ENTITY_NAME_PLURAL"), # NLUController должен будет нормализовать к единственному числу если нужно
                        "attributes": {
                            "count": int(slots.get("COUNT", 0)) if slots.get("COUNT","0").isdigit() else slots.get("COUNT"), # Попытка парсинга
                            "location": slots.get("LOCATION"),
                            "color": slots.get("ATTRIBUTE_VALUE")
                        }
                    }]
                }
            },
            {
                "intent": "QUERY_COUNT",
                "patterns": [
                    "how many <ENTITY_NAME_PLURAL> are in the <LOCATION>",
                    "how many <ENTITY_NAME_PLURAL>",
                ],
                "slots": ["ENTITY_NAME_PLURAL", "LOCATION"], "optional_slots": ["LOCATION"],
                "target_json_builder": lambda slots: {
                    "query_type": "GET_ATTRIBUTE",
                    "entity_name": slots.get("ENTITY_NAME_PLURAL"),
                    "attribute_to_query": "count",
                    "conditions": [{"location": slots.get("LOCATION")}] if slots.get("LOCATION") else []
                }
            },
            # Пример для более сложного сценария (логическое утверждение)
            {
                "intent": "PROCESS_LOGICAL_STATEMENT_IMPLICATION",
                "patterns": [
                    "if <CONDITION_PHRASE> then <CONSEQUENCE_PHRASE>",
                    "assume <CONDITION_PHRASE> implies <CONSEQUENCE_PHRASE>",
                ],
                "slots": ["CONDITION_PHRASE", "CONSEQUENCE_PHRASE"],
                "target_json_builder": lambda slots: {
                    "statement_type": "CONDITIONAL_RULE",
                    "condition": {"raw_text": slots.get("CONDITION_PHRASE")}, # NLU должен извлечь это как текст
                    "consequence": {"raw_text": slots.get("CONSEQUENCE_PHRASE")} # NLU извлекает текст
                    # Дальнейший парсинг condition/consequence в более глубокие структуры
                    # может быть задачей NLU (если обучить на это) или LogicalController.
                    # Для обучения NLU (BERT Token Classification) мы бы размечали под-слоты внутри этих фраз.
                }
            },
            {
                "intent": "ASSERT_FACT",
                "patterns": ["<CONDITION_PHRASE> is true", "it is known that <CONDITION_PHRASE>"],
                "slots": ["CONDITION_PHRASE"],
                "target_json_builder": lambda slots: {
                    "statement_type": "FACT_ASSERTION",
                    "fact": {"raw_text": slots.get("CONDITION_PHRASE")},
                    "truth_value": True
                }
            },
            # Пример для математической/сценарной задачи
            # "What happens if you add 3 more elephants and a tiger to two elephants?"
            {
                "intent": "SIMULATE_SCENARIO_CHANGE",
                "patterns": [
                    "what happens if you add <COUNT_A> more <ENTITY_A_PLURAL> and a <ENTITY_B_SINGULAR> to <COUNT_C> <ENTITY_C_PLURAL>",
                    "to <COUNT_C> <ENTITY_C_PLURAL> , add <COUNT_A> <ENTITY_A_PLURAL> and <COUNT_B> <ENTITY_B_PLURAL> then what is the result",
                    "if we have <COUNT_C> <ENTITY_C_PLURAL> and then <COUNT_A> <ENTITY_A_PLURAL> arrive, what is the new count of <ENTITY_C_PLURAL_QUERY>",
                    "start with <COUNT_C> <ENTITY_C_PLURAL>. then <ACTION_VERB> <COUNT_A> <ENTITY_A_PLURAL>. what happens to <ENTITY_TARGET_QUERY>",
                    "given <COUNT_C> <ENTITY_C_PLURAL>, if <COUNT_A> <ENTITY_A_PLURAL> are <ACTION_PAST_VERB> and <COUNT_B> <ENTITY_B_PLURAL> also <ACTION_PAST_VERB>, predict the state of <ENTITY_TARGET_QUERY>"
                ],
                "slots": ["COUNT_A", "ENTITY_A_PLURAL", "ENTITY_B_SINGULAR", "COUNT_B", "ENTITY_B_PLURAL", "COUNT_C", "ENTITY_C_PLURAL", "ENTITY_C_PLURAL_QUERY", "ACTION_VERB", "ACTION_PAST_VERB", "ENTITY_TARGET_QUERY"],
                "optional_slots": ["COUNT_B", "ENTITY_B_PLURAL", "ENTITY_B_SINGULAR", "ENTITY_C_PLURAL_QUERY", "ACTION_VERB", "ACTION_PAST_VERB", "ENTITY_TARGET_QUERY"],
                "target_json_builder": lambda s: {
                    "scenario_context": {
                        "initial_entities": [
                            {"name": s.get("ENTITY_C_PLURAL"), "count": int(s.get("COUNT_C",0)) if str(s.get("COUNT_C","0")).isdigit() else s.get("COUNT_C")}
                        ],
                        "actions": [
                            {
                                "verb": s.get("ACTION_VERB") or s.get("ACTION_PAST_VERB") or "add", # Default to add
                                "normalized_verb": (s.get("ACTION_VERB") or s.get("ACTION_PAST_VERB") or "add").upper() + "_ACTION", # Placeholder normalization
                                "entities_involved": [
                                    {"name": s.get("ENTITY_A_PLURAL"), "count": int(s.get("COUNT_A",0)) if str(s.get("COUNT_A","0")).isdigit() else s.get("COUNT_A")},
                                ]
                            }
                        ]
                    },
                    "query_type": "PREDICT_OUTCOME",
                    "query_details": {"target_entity": s.get("ENTITY_TARGET_QUERY") or s.get("ENTITY_C_PLURAL_QUERY") or s.get("ENTITY_A_PLURAL") }
                }
            },
            {
                "intent": "PROCESS_LOGICAL_STATEMENT_IMPLICATION",
                "patterns": [
                    "if <CONDITION_PHRASE_A> then <CONSEQUENCE_PHRASE_A>",
                    "assume <CONDITION_PHRASE_A> implies <CONSEQUENCE_PHRASE_A>",
                    "given that <CONDITION_PHRASE_A> is true, it follows that <CONSEQUENCE_PHRASE_A>",
                    "when <CONDITION_PHRASE_A> and <CONDITION_PHRASE_B>, then <CONSEQUENCE_PHRASE_A>",
                    "if <CONDITION_PHRASE_A> or <CONDITION_PHRASE_B>, then <CONSEQUENCE_PHRASE_A> unless <CONDITION_PHRASE_C>",
                ],
                "slots": ["CONDITION_PHRASE_A", "CONSEQUENCE_PHRASE_A", "CONDITION_PHRASE_B", "CONDITION_PHRASE_C"],
                "optional_slots": ["CONDITION_PHRASE_B", "CONDITION_PHRASE_C"],
                "target_json_builder": lambda slots: {
                    "statement_type": "CONDITIONAL_RULE",
                    "conditions": [
                        {"raw_text": slots.get("CONDITION_PHRASE_A")},
                        {"raw_text": slots.get("CONDITION_PHRASE_B")} if slots.get("CONDITION_PHRASE_B") else None,
                        {"raw_text": slots.get("CONDITION_PHRASE_C"), "type": "exception"} if slots.get("CONDITION_PHRASE_C") else None,
                    ],
                    "consequence": {"raw_text": slots.get("CONSEQUENCE_PHRASE_A")},
                    "logical_operator": "AND" if slots.get("CONDITION_PHRASE_B") and " and " in slots.get("pattern_text","") else \
                                       "OR" if slots.get("CONDITION_PHRASE_B") and " or " in slots.get("pattern_text","") else "SINGLE"
                    # NLUController would need to parse raw_text further for specific entities/actions if needed
                }
            },
            {
                "intent": "PREDICT_EVENT_OUTCOME",
                "patterns": [
                    "what will happen if the <ENTITY_NAME_SINGULAR> <EVENT_TYPE_PRESENT> the <ENTITY_NAME_PLURAL>",
                    "predict the outcome if a <ENTITY_NAME_SINGULAR> <EVENT_TYPE_PRESENT> for <TIMEFRAME_VALUE> <TIMEFRAME_UNIT>",
                    "if the <ENTITY_NAME_SINGULAR> <MODALITY> <EVENT_TYPE_BASE> the <ENTITY_NAME_PLURAL>, what is the <PROBABILITY_DESCRIPTOR> of <CONSEQUENCE_PHRASE_A>",
                ],
                "slots": ["ENTITY_NAME_SINGULAR", "EVENT_TYPE_PRESENT", "EVENT_TYPE_BASE", "ENTITY_NAME_PLURAL", "TIMEFRAME_VALUE", "TIMEFRAME_UNIT", "MODALITY", "PROBABILITY_DESCRIPTOR", "CONSEQUENCE_PHRASE_A"],
                "optional_slots": ["ENTITY_NAME_PLURAL", "TIMEFRAME_VALUE", "TIMEFRAME_UNIT", "MODALITY", "PROBABILITY_DESCRIPTOR", "CONSEQUENCE_PHRASE_A"],
                "target_json_builder": lambda s: {
                    "event_prediction_request": {
                        "actor": s.get("ENTITY_NAME_SINGULAR"),
                        "action": s.get("EVENT_TYPE_PRESENT") or s.get("EVENT_TYPE_BASE"),
                        "target": s.get("ENTITY_NAME_PLURAL"),
                        "timeframe": {"value": s.get("TIMEFRAME_VALUE"), "unit": s.get("TIMEFRAME_UNIT")} if s.get("TIMEFRAME_VALUE") else None,
                        "modality": s.get("MODALITY"),
                        "desired_outcome_description": s.get("CONSEQUENCE_PHRASE_A"),
                        "probability_focus": s.get("PROBABILITY_DESCRIPTOR")
                    }
                }
            },
            {
                "intent": "HYPOTHETICAL_QUERY",
                "patterns": [
                    "how can I <ACTION_VERB> a <ENTITY_NAME_SINGULAR> across the <LOCATION>",
                    "what if we want to <ACTION_VERB> <COUNT> <ENTITY_NAME_PLURAL> using only a <RESOURCE_SINGULAR>",
                    "is it possible to <ACTION_VERB> <ENTITY_A_PLURAL> and <ENTITY_B_PLURAL> if we have <CONSTRAINT_PHRASE>",
                ],
                "slots": ["ACTION_VERB", "ENTITY_NAME_SINGULAR", "LOCATION", "COUNT", "ENTITY_NAME_PLURAL", "RESOURCE_SINGULAR", "ENTITY_A_PLURAL", "ENTITY_B_PLURAL", "CONSTRAINT_PHRASE"],
                "optional_slots": ["LOCATION", "COUNT", "ENTITY_NAME_PLURAL", "RESOURCE_SINGULAR", "ENTITY_A_PLURAL", "ENTITY_B_PLURAL", "CONSTRAINT_PHRASE"],
                "target_json_builder": lambda s: {
                    "query_type": "FEASIBILITY_CHECK" if "possible" in s.get("pattern_text","").lower() else "PLANNING_REQUEST",
                    "goal": {
                        "action": s.get("ACTION_VERB"),
                        "entities": [
                            s.get("ENTITY_NAME_SINGULAR"), s.get("ENTITY_NAME_PLURAL"),
                            s.get("ENTITY_A_PLURAL"), s.get("ENTITY_B_PLURAL")
                        ],
                        "target_location": s.get("LOCATION"),
                        "count": s.get("COUNT")
                    },
                    "constraints": [
                        {"type": "resource", "value": s.get("RESOURCE_SINGULAR")} if s.get("RESOURCE_SINGULAR") else None,
                        {"type": "general", "description": s.get("CONSTRAINT_PHRASE")} if s.get("CONSTRAINT_PHRASE") else None,
                    ]
                }
            },
            { "intent": "GENERAL_GREETING", "patterns": ["hello", "hi", "good morning", "hey there"], "slots": [], "target_json_builder": lambda s: {"type": "greeting"} },
            { "intent": "GENERAL_GOODBYE", "patterns": ["bye", "goodbye", "see you later", "talk to you soon"], "slots": [], "target_json_builder": lambda s: {"type": "farewell"} },
            { "intent": "OUT_OF_SCOPE", "patterns": ["what is the weather today", "tell me a joke", "sing a song"], "slots": [], "target_json_builder": lambda s: {"type": "out_of_scope", "original_query": s.get("pattern_text")} },
            # Add more intents and patterns here following the structure.
            # Remember to add corresponding slot fillers in _get_slot_fillers
        ]
        # Add a 'comment' field or similar to your intent configs if you want to provide explanations
        # directly within the configuration structure.
        # For example:
        # {
        #   "intent": "EXAMPLE_INTENT",
        #   "comment": "This intent is for demonstrating how to add comments.",
        #   "patterns": [...],
        #   ...
        # }
        return config

    def _get_slot_fillers(self) -> Dict[str, List[str]]:
        """
        Defines example values for each slot type.
        For robust training, these lists should be greatly expanded with diverse examples.
        """
        fillers = {
            "ENTITY_NAME_PLURAL": ["apples", "books", "chairs", "cars", "dogs", "cats", "tables", "elephants", "tigers", "solutions", "problems"],
            "ENTITY_NAME_SINGULAR": ["apple", "book", "chair", "car", "dog", "cat", "table", "elephant", "tiger", "solution", "problem", "river"],
            "ENTITY_A_PLURAL": ["elephants", "apples", "cars", "issues", "tasks"],
            "ENTITY_B_SINGULAR": ["tiger", "banana", "driver", "key", "manager"],
            "ENTITY_B_PLURAL": ["tigers", "bananas", "drivers", "keys", "managers"],
            "ENTITY_C_PLURAL": ["elephants", "books", "people", "items", "resources"],
            "ENTITY_C_PLURAL_QUERY": ["elephants", "books", "items"], # For queries within simulations
            "ENTITY_TARGET_QUERY": ["elephants", "the items", "resource status"],
            "LOCATION": ["room", "box", "table", "garden", "garage", "shelf", "kitchen", "jungle enclosure", "database", "system", "river bank"],
            "COUNT": ["2", "5", "1", "3", "10", "a couple of", "several", "one", "two", "zero", "many", "few", "no"],
            "COUNT_A": ["3", "two", "five", "another", "some"],
            "COUNT_B": ["1", "a single", "one", "another one"],
            "COUNT_C": ["two", "five", "ten", "an initial group of", "existing"],
            "ATTRIBUTE_VALUE": ["red", "blue", "large", "green", "small", "aggressive", "friendly", "heavy", "lightweight"],
            "ATTRIBUTE_TYPE": ["color", "size", "temperament", "weight", "status"],
            "EVENT_TYPE_PRESENT": ["attacks", "eats", "moves", "hides", "joins", "leaves"],
            "EVENT_TYPE_BASE": ["attack", "eat", "move", "hide", "join", "leave"],
            "ACTION_VERB": ["add", "remove", "transport", "solve", "achieve", "cross"],
            "ACTION_PAST_VERB": ["added", "removed", "transported", "solved", "achieved", "crossed"],
            "TIMEFRAME_VALUE": ["1", "2", "few", "several", "next", "coming"],
            "TIMEFRAME_UNIT": ["hour", "hours", "day", "days", "minute", "minutes", "week", "weeks", "month"],
            "PROBABILITY_VALUE": ["0.5", "0.8", "high", "low", "certain", "unlikely", "possible"],
            "PROBABILITY_DESCRIPTOR": ["chance", "probability", "likelihood", "risk"],
            "MODALITY": ["might", "could", "will possibly", "may", "should", "is expected to"],
            "CONDITION_PHRASE_A": ["it rains heavily", "the inventory is low", "all prerequisites are met", "the user is authenticated", "the system is stable"],
            "CONDITION_PHRASE_B": ["supply chain is disrupted", "demand increases", "security protocols are active", "the light is green"],
            "CONDITION_PHRASE_C": ["emergency protocols are triggered", "the bridge is out", "user has override permissions"],
            "CONSEQUENCE_PHRASE_A": ["the ground gets wet", "we need to reorder", "the project can start", "access is granted", "operations continue normally"],
            "RESOURCE_SINGULAR": ["boat", "truck", "budget", "team member", "tool"],
            "CONSTRAINT_PHRASE": ["the boat can only carry one item", "we have limited time", "budget is tight", "X must happen before Y"]
        }
        fillers["ENTITY_NAME"] = fillers["ENTITY_NAME_SINGULAR"] + fillers["ENTITY_NAME_PLURAL"]
        # Ensure all slot keys used in intents_config are defined here.
        # It's good practice to add comments explaining what each slot type represents.
        return fillers

    def _generate_sentence_and_slot_mentions(self, pattern_text: str, current_intent_slots: List[str], optional_slots: List[str]) \
            -> Tuple[str, List[Dict[str, Any]], Dict[str, str]]:
        sentence = pattern_text
        slot_mentions = []
        filled_slots_map: Dict[str, str] = {"pattern_text": pattern_text} # Store original pattern for context if needed by builder

        active_slots = [s for s in current_intent_slots if isinstance(s, str)]
        placeholders_in_pattern = re.findall(r"(<\w+>)", pattern)

        placeholder_to_value_map: Dict[str, str] = {}

        for placeholder_tag in placeholders_in_pattern:
            slot_type = placeholder_tag.strip("<>").upper() # Убедимся, что слот в верхнем регистре, как в fillers

            if slot_type not in active_slots:
                sentence = sentence.replace(placeholder_tag, "", 1)
                continue

            is_optional = slot_type in optional_slots
            if is_optional and random.choice([True, False]):
                sentence = sentence.replace(placeholder_tag, "", 1)
                continue

            if slot_type in self.slot_fillers and self.slot_fillers[slot_type]:
                slot_value = random.choice(self.slot_fillers[slot_type])
                placeholder_to_value_map[placeholder_tag] = slot_value
                filled_slots_map[slot_type] = slot_value
            else:
                print(f"Warning: No filler for slot_type '{slot_type}' (from placeholder '{placeholder_tag}') in pattern '{pattern}'. Skipping.")
                sentence = sentence.replace(placeholder_tag, "", 1)

        final_sentence_parts = []
        current_char_pos = 0
        last_pattern_idx = 0

        sorted_placeholders = sorted(
            [ph for ph in placeholders_in_pattern if ph in placeholder_to_value_map],
            key=lambda ph_tag: pattern.find(ph_tag)
        )

        for placeholder_tag in sorted_placeholders:
            slot_value = placeholder_to_value_map[placeholder_tag]
            slot_type = placeholder_tag.strip("<>").upper() # И здесь тоже

            ph_start_in_pattern = pattern.find(placeholder_tag, last_pattern_idx)

            if ph_start_in_pattern != -1:
                text_before = pattern[last_pattern_idx:ph_start_in_pattern]
                final_sentence_parts.append(text_before)
                current_char_pos += len(text_before)

                slot_start_char = current_char_pos
                final_sentence_parts.append(slot_value)
                current_char_pos += len(slot_value)
                slot_end_char = current_char_pos

                slot_mentions.append({
                    "text": slot_value, "type": slot_type,
                    "start_char": slot_start_char, "end_char": slot_end_char
                })
                last_pattern_idx = ph_start_in_pattern + len(placeholder_tag)
            else:
                 print(f"Warning: Placeholder '{placeholder_tag}' not found in pattern segment during mention generation.")

        final_sentence_parts.append(pattern[last_pattern_idx:])
        sentence = "".join(final_sentence_parts)
        sentence = re.sub(r'\s{2,}', ' ', sentence).strip()

        return sentence, slot_mentions, filled_slots_map


    def _get_iob_tags(self, text_sentence: str, slot_mentions: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        # ... (логика без изменений) ...
        if not self.tokenizer:
            tokens = text_sentence.split()
            iob_tags = ['O'] * len(tokens)
            return tokens, iob_tags

        tokenized_output = self.tokenizer.encode_plus(
            text_sentence, return_offsets_mapping=True, add_special_tokens=True,
            max_length=512, truncation=True )

        tokens_from_tokenizer = self.tokenizer.convert_ids_to_tokens(tokenized_output["input_ids"])
        offset_mapping = tokenized_output["offset_mapping"]
        iob_tags = ['O'] * len(tokens_from_tokenizer)

        for mention in slot_mentions:
            mention_start_char = mention["start_char"]
            mention_end_char = mention["end_char"]
            slot_type = mention["type"]
            first_token_in_slot = True
            for i, (offset_start, offset_end) in enumerate(offset_mapping):
                if offset_end == 0 and offset_start == 0: continue
                if max(mention_start_char, offset_start) < min(mention_end_char, offset_end):
                    if first_token_in_slot:
                        iob_tags[i] = f"B-{slot_type}"
                        first_token_in_slot = False
                    else: iob_tags[i] = f"I-{slot_type}"
        return tokens_from_tokenizer, iob_tags

    def generate_sample(self) -> Optional[Dict[str, Any]]:
        if not self.tokenizer:
            print("ERROR: Tokenizer not available in generate_sample.")
            return None

        intent_config = random.choice(self.intents_config)
        intent_label = intent_config["intent"]
        pattern = random.choice(intent_config["patterns"])

        current_intent_slots = intent_config.get("slots", [])
        optional_slots = intent_config.get("optional_slots", [])

        text_sentence, slot_mentions, filled_slots_map = self._generate_sentence_and_slot_mentions(
            pattern, current_intent_slots, optional_slots
        )

        if not text_sentence.strip(): return None

        tokens, iob_tags = self._get_iob_tags(text_sentence, slot_mentions)

        if not tokens or len(tokens) != len(iob_tags) : return None

        # Генерируем целевой структурированный JSON
        target_json_builder = intent_config.get("target_json_builder")
        target_structured_data = {}
        if target_json_builder:
            try:
                target_structured_data = target_json_builder(filled_slots_map)
            except Exception as e:
                print(f"Error building target_structured_output for intent {intent_label} with slots {filled_slots_map}: {e}")
                target_structured_data = {"error_building_json": str(e)}
        else: # Если нет builder'а, просто складываем извлеченные слоты (старое поведение)
             target_structured_data = {"intent": intent_label, "extracted_slots": filled_slots_map}


        return {
            "text": text_sentence,
            "intent": intent_label, # Это будет целью для Intent Classification
            "tokens": tokens,
            "iob_tags": iob_tags, # Это будет целью для Token Classification (Slot Filling)
            "target_structured_output": target_structured_data # Это эталонный JSON для всего NLU пайплайна
        }

    def generate_data(self, num_samples: int) -> List[Dict[str, Any]]:
        # ... (логика без изменений) ...
        data = []
        generated_count = 0
        attempts = 0
        max_attempts = num_samples * 5

        while generated_count < num_samples and attempts < max_attempts:
            sample = self.generate_sample()
            attempts += 1
            if sample:
                data.append(sample)
                generated_count += 1

        if generated_count < num_samples:
            print(f"Warning: Generated only {generated_count}/{num_samples} samples after {max_attempts} attempts.")
        return data

    def save_data_to_jsonl(self, data: List[Dict[str, Any]], filepath: str):
        # ... (логика без изменений) ...
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Data successfully saved to {filepath}")
        except IOError as e:
            print(f"Error saving data to {filepath}: {e}")


if __name__ == '__main__':
    print("Initializing TrainingDataGenerator (English)...")
    generator = TrainingDataGenerator(bert_tokenizer_name=DEFAULT_TOKENIZER_NAME)

    if generator.tokenizer is None:
        print("Stopping script because tokenizer could not be loaded.")
    else:
        num_generated_samples = 20 # Уменьшим для быстрого теста изменений
        print(f"\nGenerating {num_generated_samples} samples with new complex structures...")
        generated_data = generator.generate_data(num_generated_samples)

        if generated_data:
            print(f"\n--- Example of {min(5, len(generated_data))} Generated Samples (with target_structured_output) ---")
            for i, sample in enumerate(generated_data[:5]):
                print(f"\nSample {i+1}:")
                print(f"  Text: {sample['text']}")
                print(f"  Intent (for BERT training): {sample['intent']}")
                # print(f"  Tokens: {sample['tokens']}")
                # print(f"  IOB Tags (for BERT training): {sample['iob_tags']}")
                print(f"  Target Structured Output (for NLUController post-processing & eval): {json.dumps(sample['target_structured_output'], indent=2)}")
                if len(sample['tokens']) != len(sample['iob_tags']):
                    print(f"  CRITICAL WARNING: Tokens length != IOB Tags length!")

            output_directory = "data"
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok=True)

            output_filepath = os.path.join(output_directory, "synthetic_nlu_data_en_complex.jsonl")
            print(f"\nSaving generated data to {output_filepath}...")
            generator.save_data_to_jsonl(generated_data, output_filepath)
            print(f"Total samples generated: {len(generated_data)}")
        else:
            print("No data was generated. Check scenarios definition in _define_scenarios().")

    print("\nTrainingDataGenerator script finished.")
