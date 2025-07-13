import json
import random
from typing import List, Dict, Any, Tuple

"""
This module is responsible for generating training data for the CommandNetwork.
The goal is to create pairs of (natural language text, corresponding command sequence).
This is a foundational piece of the project and will need to be significantly
expanded with a rich variety of templates and logic.
"""

class CommandNetworkDataGenerator:
    """
    Generates synthetic data for training the CommandNetwork.
    """
    def __init__(self):
        # In a real scenario, this would be a very large and complex structure.
        self.templates = self._get_initial_templates()
        self.entity_map = {
            "стула": "стул", "яблока": "яблоко", "карандаша": "карандаш",
            "слонам": "слон", "тигра": "тигр", "слонов": "слон"
        }
        self.count_map = {
            "два": 2, "3": 3, "пять": 5, "двум": 2, "трех": 3
        }


    def _get_initial_templates(self) -> List[Dict[str, Any]]:
        """
        Defines the templates for data generation.
        Each template contains:
        - A sentence pattern with placeholders.
        - The logic to generate the corresponding command sequence.
        """
        templates = [
            {
                "name": "Create single object with count",
                "pattern": "в <LOCATION> есть <COUNT> <ENTITY_PLURAL>",
                "placeholders": {
                    "<LOCATION>": ["комнате", "коробке", "саду"],
                    "<COUNT>": ["два", "3", "пять"],
                    "<ENTITY_PLURAL>": ["стула", "яблока", "карандаша"]
                },
                "command_generator": self._generate_create_object_commands
            },
            {
                "name": "Complex scenario: create, update, query",
                "pattern": "К <INITIAL_COUNT_PHRASE> <ENTITY_A_PLURAL> добавили еще <COUNT_TO_ADD_PHRASE>, а потом <ENTITY_B_SINGULAR>. Что будет?",
                "placeholders": {
                    "<INITIAL_COUNT_PHRASE>": ["двум", "пяти"],
                    "<ENTITY_A_PLURAL>": ["слонам"],
                    "<COUNT_TO_ADD_PHRASE>": ["трех", "двух"],
                    "<ENTITY_B_SINGULAR>": ["тигра"]
                },
                "command_generator": self._generate_complex_scenario_commands
            },
        ]
        return templates

    def _get_normalized_entity(self, text: str) -> str:
        return self.entity_map.get(text, text)

    def _get_normalized_count(self, text: str) -> int:
        # Extend the map to include all cases from templates
        full_count_map = {
            "два": 2, "3": 3, "пять": 5, "двум": 2, "трех": 3,
            "двух": 2, "пяти": 5
        }
        return full_count_map.get(text, int(text) if text.isdigit() else 1)


    def _generate_create_object_commands(self, filled: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Generates commands for a "create object" sentence.
        """
        count = self._get_normalized_count(filled["<COUNT>"])
        entity_name = self._get_normalized_entity(filled["<ENTITY_PLURAL>"])
        location = self._get_normalized_entity(filled["<LOCATION>"])

        obj_id = f"{entity_name}_1"
        loc_id = f"{location}_1"

        return [
            {"action": "CREATE_NODE", "node_type": "Object", "data": {"name": entity_name, "state": "REAL", "attributes": {"count": count}, "node_id": obj_id}},
            {"action": "CREATE_NODE", "node_type": "Object", "data": {"name": location, "state": "REAL", "attributes": {}, "node_id": loc_id}},
            {"action": "CREATE_EDGE", "source_id": obj_id, "target_id": loc_id, "edge_type": "LOCATED_IN"}
        ]

    def _generate_complex_scenario_commands(self, filled: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Generates commands for the complex "elephants and tiger" scenario.
        """
        initial_count = self._get_normalized_count(filled["<INITIAL_COUNT_PHRASE>"])
        entity_a_name = self._get_normalized_entity(filled["<ENTITY_A_PLURAL>"])
        count_to_add = self._get_normalized_count(filled["<COUNT_TO_ADD_PHRASE>"])
        entity_b_name = self._get_normalized_entity(filled["<ENTITY_B_SINGULAR>"])

        entity_a_id = f"{entity_a_name}_1"
        entity_b_id = f"{entity_b_name}_1"
        query_id = "query_1"

        return [
            {
                "action": "CREATE_NODE",
                "node_type": "Object",
                "data": {"name": entity_a_name, "state": "REAL", "attributes": {"count": initial_count}, "node_id": entity_a_id}
            },
            {
                "action": "UPDATE_NODE_PROPERTY",
                "node_id": entity_a_id,
                "property_key": "count",
                "property_value": f"PREVIOUS_VALUE + {count_to_add}"
            },
            {
                "action": "CREATE_NODE",
                "node_type": "Object",
                "data": {"name": entity_b_name, "state": "HYPOTHETICAL", "attributes": {"count": 1}, "node_id": entity_b_id}
            },
            {
               "action": "CREATE_EDGE",
               "source_id": entity_b_id,
               "target_id": entity_a_id,
               "edge_type": "IN_SAME_SCENARIO_AS"
            },
            {
               "action": "CREATE_NODE",
               "node_type": "Object",
               "data": {
                   "name": "world_state_query",
                   "state": "QUERY",
                   "node_id": query_id,
                   "formula": {'outcome': f'FUNC::PREDICT({entity_a_id}, {entity_b_id})'}
               }
           }
        ]

    def generate_sample(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generates a single training sample (text, commands).
        """
        template = random.choice(self.templates)

        sentence = template["pattern"]
        filled_placeholders = {}

        for placeholder, values in template["placeholders"].items():
            chosen_value = random.choice(values)
            sentence = sentence.replace(placeholder, chosen_value, 1)
            filled_placeholders[placeholder] = chosen_value

        # The command generator method is called via the instance, so `self` is passed implicitly.
        commands = template["command_generator"](filled_placeholders)

        return sentence, commands

    def generate_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Generates a list of training samples.
        """
        data = []
        for _ in range(num_samples):
            text, commands = self.generate_sample()
            data.append({"text": text, "commands": commands})
        return data


if __name__ == '__main__':
    generator = CommandNetworkDataGenerator()
    print("--- Generating Training Data for CommandNetwork ---")

    # Generate more samples to see both templates in action
    generated_data = generator.generate_data(10)

    for i, sample in enumerate(generated_data):
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {sample['text']}")
        print("Generated Commands:")
        print(json.dumps(sample['commands'], indent=2, ensure_ascii=False))

    output_filename = "command_network_training_data.jsonl"
    print(f"\nSaving generated data to {output_filename}...")
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            for item in generated_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Successfully saved {len(generated_data)} samples.")
    except IOError as e:
        print(f"Error saving file: {e}")

    print("\n--- Data Generation Finished ---")
