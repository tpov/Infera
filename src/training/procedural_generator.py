import random
import json

class ProceduralGenerator:
    def __init__(self):
        print("Initialized ProceduralGenerator.")

    def generate_arithmetic_task(self, max_ops=2):
        """
        Generates a single arithmetic task.
        Example: 5 + 8 * 2
        """
        ops = ['+', '-', '*']
        num_ops = random.randint(1, max_ops)

        expression = [str(random.randint(1, 20))]
        for _ in range(num_ops):
            expression.append(random.choice(ops))
            expression.append(str(random.randint(1, 10)))

        expression_str = " ".join(expression)

        try:
            result = eval(expression_str)
        except ZeroDivisionError:
            return None

        word_map = {'+': 'плюс', '-': 'минус', '*': 'умножить на'}
        nl_expression = expression_str
        for op, word in word_map.items():
            nl_expression = nl_expression.replace(op, word)
        natural_language_question = f"сколько будет {nl_expression}"

        declarations = []
        node_ids = []
        for i, token in enumerate(expression):
            if token.isdigit():
                node_id = f"число_{token}_{i}"
                node_ids.append(node_id)
                declarations.append({
                    "type": "Object", "domain": "CONCEPTUAL",
                    "name": node_id, "value": int(token)
                })

        # This simplified logic doesn't create a proper execution tree,
        # but declares all necessary components for the controller to work with.
        op_count = 0
        for token in expression:
            if token in ops:
                op_count += 1
                declarations.append({ "type": "Intent", "name": f"операция_{op_count}", "goal": f"выполнить {token}" })

        declarations.append({
            "type": "Object", "name": "финальный_вопрос", "domain": "CONCEPTUAL",
            "is_query": True, "description": f"вычислить выражение: {expression_str}",
            "dependencies": node_ids # The query depends on all number objects
        })

        return {"text": natural_language_question, "declarations": declarations}

    def generate_logical_task(self):
        """Generates a single boolean logic task."""
        # To be implemented
        pass

    def generate_dataset(self, num_samples: int):
        dataset = []
        for i in range(num_samples):
            if i % 2 == 0:
                task = self.generate_arithmetic_task()
            else:
                task = self.generate_logical_task()

            if task:
                dataset.append(task)
        return dataset

if __name__ == '__main__':
    generator = ProceduralGenerator()
    print("--- Generating Procedural Training Data ---")

    dataset = generator.generate_dataset(20) # Generate 20 mixed samples

    for i, sample in enumerate(dataset):
        # A simple way to check task type for logging
        task_type = "Arithmetic" if "сколько" in sample['text'] else "Logical"
        print(f"\n--- Sample {i+1} (Type: {task_type}) ---")
        print(f"Text: {sample['text']}")
        print("Generated Declarations:", json.dumps(sample['declarations'], indent=2, ensure_ascii=False))

    output_filename = "procedural_training_data.jsonl"
    print(f"\nSaving generated data to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Successfully saved {len(dataset)} samples.")
    print("\n--- Data Generation Finished ---")
