import json
import random
import os
from typing import List, Dict, Any, Optional, Tuple

THINKING_OUTPUT_SEPARATOR = "%%IDEAS:%%"

class NlgTrainingDataGenerator:
    """
    Генерирует синтетические данные для обучения NLG модели (например, BART).
    Создает пары (входная_строка_для_NLG, целевой_текст_ответа).
    """
    def __init__(self):
        self.scenarios = self._define_scenarios()
        print("NlgTrainingDataGenerator initialized.")

    def _format_nlg_input(self, task_type: str, nlu_result_for_nlg: Dict[str, Any],
                          state_for_nlg: Dict[str, Any]) -> str:
        """
        Форматирует структурированные данные в одну строку для подачи на вход NLG модели.
        nlu_result_for_nlg может содержать 'intent', 'slots', а также 'meta_missing_slot', 'meta_prompt_info'.
        state_for_nlg содержит 'entities', 'facts', 'active_goals'.
        """
        intent_str = nlu_result_for_nlg.get("overall_intent", nlu_result_for_nlg.get("intent", "none")) # Учитываем оба варианта ключа

        # NLUController теперь возвращает сложный JSON, где "сырые" слоты от BERT могут быть в "slots_extracted_by_bert"
        # или уже разобраны в более специфичные поля.
        # Для _format_nlg_input, нам нужен стандартизированный способ получения списка слотов.
        # Будем ожидать, что nlu_result_for_nlg содержит ключ "slots" со списком словарей {type: value}
        # или "slots_extracted_by_bert" если это "сырой" вывод NLUController до полной сборки.
        slots_list = nlu_result_for_nlg.get("slots", nlu_result_for_nlg.get("slots_extracted_by_bert", []))
        if not isinstance(slots_list, list): slots_list = []

        slots_parts = [f"{s.get('type','unk_type')}:{s.get('value','unk_val')}" for s in slots_list]
        slots_str = ", ".join(slots_parts) if slots_parts else "none"

        entities_parts = []
        for entity_name, attrs in state_for_nlg.get("entities", {}).items():
            attr_strs = [f"{name}:{val}" for name, val in attrs.items()]
            entities_parts.append(f"{entity_name}({', '.join(attr_strs)})")
        entities_str = ", ".join(entities_parts) if entities_parts else "none"

        facts_parts = []
        for fact in state_for_nlg.get("facts", []):
            if isinstance(fact, dict):
                fact_detail_parts = [f"{k}:{str(v)}" for k,v in fact.items()]
                facts_parts.append(f"fact({', '.join(fact_detail_parts)})")
            elif isinstance(fact, str):
                facts_parts.append(f"fact({fact})")
        facts_str = ", ".join(facts_parts) if facts_parts else "none"

        goals_parts = []
        for goal in state_for_nlg.get("active_goals", []):
            goal_desc_parts = [f"{k}:{str(v)}" for k,v in goal.items()]
            goals_parts.append(f"goal({', '.join(goal_desc_parts)})")
        goals_str = ", ".join(goals_parts) if goals_parts else "none"

        meta_info_parts = []
        # meta_missing_slot может быть в nlu_result_for_nlg, если LogicalController его добавил
        if nlu_result_for_nlg.get("meta_missing_slot"):
            meta_info_parts.append(f"missing_focus:{nlu_result_for_nlg['meta_missing_slot']}")
        # meta_prompt_info может быть в nlu_result_for_nlg (если передан из kwargs в NLGModel.generate_text)
        if nlu_result_for_nlg.get("meta_prompt_info"):
             meta_info_parts.append(f"prompt_hint:{nlu_result_for_nlg['meta_prompt_info']}")
        meta_info_str = " | ".join(meta_info_parts) if meta_info_parts else "none"

        input_text = f"task: {task_type} | intent: {intent_str} | slots: {slots_str} | meta: {meta_info_str} | entities: {entities_str} | facts: {facts_str} | goals: {goals_str}"
        return input_text

    def _get_slot_fillers(self) -> Dict[str, List[str]]:
        return {
            "ENTITY_NAME": ["apples", "books", "tigers", "elephants", "business plan draft", "marketing strategy", "budget proposal"],
            "LOCATION": ["room", "box", "table", "jungle", "market", "online store"],
            "COUNT": ["2", "5", "1", "3", "a few", "ten", "many"],
            "ATTRIBUTE_VALUE": ["red", "large", "aggressive", "detailed", "comprehensive", "approved"],
            "ATTRIBUTE_TYPE": ["color", "size", "temperament", "status", "complexity"],
            "ITEM": ["budget", "marketing plan", "supplier list", "user feedback", "risk_assessment"],
            "STATUS": ["pending", "approved", "rejected", "in_progress", "completed", "high_priority"],
            "REASON": ["lack of funding", "positive market demand", "strong_competition", "strategic_alignment_needed"],
            "ACTOR": ["tiger", "user", "competitor", "investor", "marketing_team"],
            "TARGET": ["elephant", "business_plan", "market_segment", "product_launch"],
            "EVENT_TYPE": ["attack", "define_scope", "analyze_market", "mitigate_risk", "launch_product"],
            "TIMEFRAME_VALUE": ["1", "next_week", "short_term", "three_months", "end_of_year"],
            "TIMEFRAME_UNIT": ["hour", "days", "phase", "quarter"],
            "PROBABILITY_VALUE": ["0.7", "high", "medium", "low", "0.25"],
            "PROBABILITY_DESCRIPTOR": ["chance", "probability", "risk_level", "likelihood_score"],
            "MODALITY": ["might", "will likely", "should consider", "must complete", "could potentially"],
            "FIELD": ["online_store", "eco_tourism", "software_development", "consulting_services"],
            "USER_SKILL": ["programming", "marketing", "financial_planning", "graphic_design", "project_management"],
            "USER_EXPERIENCE": ["software_development_5_years", "startup_founder", "no_previous_business_experience", "marketing_manager_3_years"]
        }

    def _define_scenarios(self) -> List[Dict[str, Any]]:
        scenarios = []
        sf = self._get_slot_fillers() # sf для краткости

        # --- GENERATE_RESPONSE ---
        scenarios.append({
            "description": "Ответ на запрос количества с учетом выведенного факта об угрозе",
            "nlu_result": {"overall_intent": "SIMULATE_SCENARIO_CHANGE_RESULT",
                           "slots_extracted_by_bert": [
                                     {"type": "ENTITY_NAME", "value": sf["ENTITY_NAME"][2]}, # tigers
                                     {"type": "COUNT", "value": sf["COUNT"][2]}, # 1
                                     {"type": "ENTITY_NAME", "value": sf["ENTITY_NAME"][3]}, # elephants
                                     {"type": "COUNT", "value": sf["COUNT"][1]}, # 5
                                    ]},
            "state": {
                "entities": {sf["ENTITY_NAME"][3]: {"count": 5}, sf["ENTITY_NAME"][2]: {"count": 1, "temperament": sf["ATTRIBUTE_VALUE"][2]}},
                "facts": [{"subject": sf["ENTITY_NAME"][2], "predicate": "IS_A", "object": "carnivore", "type": "kb_derived"},
                          {"subject": sf["ENTITY_NAME"][2], "predicate": "CAN_EAT", "object": sf["ENTITY_NAME"][3], "type": "inferred"}],
                "active_goals": [] },
            "task_type": "GENERATE_RESPONSE",
            "target_texts": [
                f"There will be {sf['COUNT'][1]} {sf['ENTITY_NAME'][3]} and {sf['COUNT'][2]} {sf['ENTITY_NAME'][2]}. Given that {sf['ENTITY_NAME'][2]} are carnivores and can eat {sf['ENTITY_NAME'][3]}, caution is advised.",
            ]
        })

        # --- REQUEST_CLARIFICATION ---
        scenarios.append({
            "description": "Запрос на уточнение бюджета для бизнес-плана, с meta_missing_slot",
            "nlu_result": {"overall_intent": "QUERY_BUSINESS_PLAN",
                           "slots_extracted_by_bert": [{"type": "FIELD", "value": sf["FIELD"][0]}],
                           "meta_missing_slot": "budget_estimation"},
            "state": { "entities": {"user_profile": {"experience": sf["USER_EXPERIENCE"][0]}}, "facts": [], "active_goals": [] },
            "task_type": "REQUEST_CLARIFICATION",
            "target_texts": [
                f"To plan the {sf['FIELD'][0]} business, I need the budget_estimation. What is your approximate budget?",
            ]
        })

        # --- THINK_AND_PLAN ---
        scenarios.append({
            "description": "Начало 'думания' о бизнес-плане",
            "nlu_result": {"overall_intent": "QUERY_BUSINESS_PLAN", "slots_extracted_by_bert": [{"type": "FIELD", "value": "gourmet cat food delivery"}]},
            "state": {
                "entities": {"user_profile": {"experience": sf["USER_EXPERIENCE"][1], "skills": sf["USER_SKILL"][1]}},
                "facts": [{"subject": "gourmet_cat_food", "predicate": "HAS_DEMAND_IN", "object": "urban_areas", "type":"kb_derived"}],
                "active_goals": [{"goal_id": "bp_cat_food", "goal_type": "PLAN_BUSINESS", "status": "thinking_initiated",
                                  "missing_info": ["budget", "target_audience"]}]
            },
            "task_type": "THINK_AND_PLAN",
            "target_texts": [
                f"Let's create a plan for 'gourmet cat food delivery'. We know there's demand in urban areas. What's your budget and who is the target audience? {THINKING_OUTPUT_SEPARATOR} Idea: Analyze existing pet food delivery services. Suggestion: Define unique selling points like organic ingredients or fast delivery. Question: What are the initial marketing channels you are considering?",
            ]
        })

        # --- EXPLAIN_REASONING ---
        scenarios.append({
            "description": "Объяснение выбора стратегии с использованием meta_prompt_info",
            "nlu_result": {"overall_intent": "INTERNAL_QUERY_REASONING",
                           "slots_extracted_by_bert": [{"type":"STRATEGY_ID", "value":"S002"}],
                           "meta_prompt_info": "Explain choice of S002 focusing on user skills."},
            "state": { "entities": { "user_profile": {"skills": sf["USER_SKILL"][2]}, # financial_planning
                                     "business_plan_draft": {"strategy_S002_details": "Bootstrap with minimal initial marketing, focus on organic growth."}},
                       "facts": [{"subject": "bootstrapping", "predicate": "SUITS_SKILL", "object": sf["USER_SKILL"][2], "type":"inferred"}],
                       "active_goals": [] },
            "task_type": "EXPLAIN_REASONING",
            "target_texts": [
                f"The bootstrapping strategy (S002) was suggested as it aligns well with your financial_planning skills, allowing for controlled organic growth with minimal initial marketing spend.",
            ]
        })
        return scenarios

    def generate_data(self, num_samples_per_target_text: int = 1) -> List[Dict[str, str]]:
        training_data = []
        for scenario in self.scenarios:
            nlu_data_for_nlg = scenario["nlu_result"]
            state_data_for_nlg = scenario["state"]
            task_type = scenario["task_type"]

            input_text = self._format_nlg_input(task_type, nlu_data_for_nlg, state_data_for_nlg)

            for target_text in scenario["target_texts"]:
                for _ in range(num_samples_per_target_text):
                    training_data.append({"input_text": input_text, "target_text": target_text})
        return training_data

    def save_data_to_jsonl(self, data: List[Dict[str, str]], filepath: str):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"NLG training data successfully saved to {filepath}")
        except IOError as e:
            print(f"Error saving NLG training data to {filepath}: {e}")

if __name__ == '__main__':
    print("Initializing NlgTrainingDataGenerator...")
    generator = NlgTrainingDataGenerator()
    print(f"\nGenerating NLG training data based on defined scenarios...")
    generated_nlg_data = generator.generate_data(num_samples_per_target_text=2)

    if generated_nlg_data:
        print(f"\n--- Example of {min(5, len(generated_nlg_data))} Generated NLG Samples ---")
        for i, sample in enumerate(generated_nlg_data[:5]):
            print(f"\nSample {i+1}:")
            print(f"  Input Text (for NLG model): {sample['input_text']}")
            print(f"  Target Text (expected NLG output): {sample['target_text']}")

        output_directory = "data"
        if not os.path.exists(output_directory): os.makedirs(output_directory, exist_ok=True)
        output_filepath = os.path.join(output_directory, "synthetic_nlg_bart_data_en.jsonl")
        print(f"\nSaving generated NLG data to {output_filepath}...")
        generator.save_data_to_jsonl(generated_nlg_data, output_filepath)
        print(f"Total samples generated: {len(generated_nlg_data)}")
    else:
        print("No NLG data was generated. Check scenarios definition in _define_scenarios().")
    print("\nNlgTrainingDataGenerator script finished.")
