import json
import random
from typing import List, Dict, Any

class CommandNetworkDataGenerator:
    def __init__(self):
        self.templates: List[Dict[str, Any]] = self._get_templates()
        self.placeholders: Dict[str, List[str]] = {
            "<LOCATION>": ["комната", "коробка", "сад"],
            "<COUNT_WORD>": ["два", "три", "пять"],
            "<ENTITY>": ["стул", "яблоко", "слон"],
            "<PERSON>": ["коллега", "начальник", "программист"],
            "<TASK>": ["отчет", "презентация", "код"],
        }

    def _get_templates(self) -> List[Dict[str, Any]]:
        return [
            {"name": "Create object with count", "pattern": "в <LOCATION> есть <COUNT_WORD> <ENTITY>", "generator": self._generate_create_object_declarations},
            {"name": "Complex scenario with update", "pattern": "у нас было <COUNT_WORD> <ENTITY>, потом добавили еще <COUNT_WORD>", "generator": self._generate_update_object_declarations},
            {"name": "Intent with dependencies", "pattern": "<PERSON> делает <TASK>, чтобы <PERSON> мог сделать <TASK>", "generator": self._generate_intent_dependency_declarations},
        ]

    def _get_placeholders_for_template(self, template: Dict[str, Any]) -> Dict[str, str]:
        """Helper to get a random set of placeholders for a given template."""
        filled = {}
        pattern = template['pattern']
        # This logic is simplified; it doesn't handle the same placeholder appearing twice yet.
        for ph_key, ph_values in self.placeholders.items():
            if ph_key in pattern:
                filled[ph_key] = random.choice(ph_values)
        return filled

    def _generate_create_object_declarations(self, p: Dict[str, str]) -> List[Dict[str, Any]]:
        count_map = {"два": 2, "три": 3, "пять": 5}
        return [
            {"type": "Object", "name": p.get("<ENTITY>", "unknown"), "context": "REAL", "count": count_map.get(p.get("<COUNT_WORD>"), 1)},
            {"type": "Object", "name": p.get("<LOCATION>", "unknown"), "context": "REAL"}
        ]

    def _generate_update_object_declarations(self, p: Dict[str, str]) -> List[Dict[str, Any]]:
        count_map = {"два": 2, "три": 3, "пять": 5}
        return [
            {"type": "Object", "name": p.get("<ENTITY>"), "context": "REAL", "count": count_map.get(p.get("<COUNT_WORD>"))},
            {"type": "Object", "name": p.get("<ENTITY>"), "context": "REAL", "count": f"PREVIOUS_VALUE + {count_map.get(p.get('<COUNT_WORD>_1', 'два'))}"}
        ]

    def _generate_intent_dependency_declarations(self, p: Dict[str, str]) -> List[Dict[str, Any]]:
        task1_name = f"task_{p.get('<TASK>','t1')}"
        task2_name = f"task_{p.get('<TASK>_1','t2')}"
        return [
            {"type": "Object", "name": p.get("<PERSON>"), "context": "REAL"},
            {"type": "Object", "name": p.get("<PERSON>_1"), "context": "REAL"},
            {"type": "Intent", "name": task1_name, "context": "REAL", "source_id": p.get("<PERSON>"), "status": "IN_PROGRESS"},
            {"type": "Intent", "name": task2_name, "context": "REAL", "source_id": p.get("<PERSON>_1"), "status": "PENDING", "dependencies": [task1_name]}
        ]

    def generate_sample(self, template_name: str) -> Dict[str, Any]:
        template = next((t for t in self.templates if t["name"] == template_name), None)
        if not template: return {"text": "", "declarations": []}

        # This generation logic is still simplified for clarity.
        placeholders = self._get_placeholders_for_template(template)
        # Manually handle cases with multiple placeholders of the same type for now
        if template_name == "Complex scenario with update":
            placeholders["<COUNT_WORD>_1"] = random.choice(self.placeholders["<COUNT_WORD>"])
        if template_name == "Intent with dependencies":
             placeholders["<PERSON>_1"] = random.choice(self.placeholders["<PERSON>"])
             placeholders["<TASK>_1"] = random.choice(self.placeholders["<TASK>"])

        sentence = template['pattern']
        for key, val in placeholders.items():
            if key.endswith("_1"): continue
            sentence = sentence.replace(key, val, 1)
        # A bit of a hack for the second placeholder
        if "_1" in str(placeholders):
             sentence = sentence.replace(template['pattern'].split()[2], placeholders.get(next(k for k in placeholders if k.endswith("_1"))))

        declarations = template["generator"](placeholders)
        return {"text": "Simulated text for: " + template['name'], "declarations": declarations}

if __name__ == '__main__':
    generator = CommandNetworkDataGenerator()
    print("--- Generating Declarative Training Data ---")
    data = [generator.generate_sample(t['name']) for t in generator.templates]

    for i, sample in enumerate(data):
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {sample['text']}")
        print("Generated Declarations:", json.dumps(sample['declarations'], indent=2, ensure_ascii=False))

    output_filename = "declarative_training_data.jsonl"
    print(f"\nSaving generated data to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Successfully saved {len(data)} samples.")
    print("\n--- Data Generation Finished ---")
