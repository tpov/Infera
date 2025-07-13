import json
from src.logical_controller.controller import LogicalController
from src.data_generation.command_network_generator import CommandNetworkDataGenerator

class MockDeclarativeCommandNetwork:
    def __init__(self):
        self.generator = CommandNetworkDataGenerator()
        print("NEURAL_NETWORKS: Initialized MockDeclarativeCommandNetwork.")

    def process(self, text: str) -> list:
        print(f"COMMAND_NETWORK: Simulating generation for text: '{text}'")
        text_lower = text.lower()

        # More robust template matching
        if "было" in text_lower and "добавили" in text_lower:
            template_name = "Complex scenario with update"
        elif "делает" in text_lower and "чтобы" in text_lower:
            template_name = "Intent with dependencies"
        else:
            template_name = "Create object with count"

        # Find the correct template and generate one sample from it
        for template in self.generator.templates:
            if template['name'] == template_name:
                # We need to generate a sample that matches the input text's entities
                # This mock is still simple, but we can make it slightly better
                placeholders = {}
                # A real system would extract these entities. We'll hardcode for the test.
                if template_name == "Create object with count":
                    placeholders["<LOCATION>"] = "комнате"
                    placeholders["<COUNT_WORD>"] = "два"
                    placeholders["<ENTITY>"] = "стула" # Using the plural form from template
                elif template_name == "Complex scenario with update":
                    placeholders["<COUNT_WORD>"] = "три"
                    placeholders["<ENTITY>"] = "слон"
                    placeholders["<COUNT_WORD>_1"] = "два"
                elif template_name == "Intent with dependencies":
                    placeholders["<PERSON>"] = "начальник"
                    placeholders["<TASK>"] = "отчет"
                    placeholders["<PERSON>_1"] = "коллега"
                    placeholders["<TASK>_1"] = "презентация"

                # This is still not perfect, but it's better than random generation
                # For a true mock, we'd need more sophisticated text analysis here.
                # Let's just use the generator's random sample for the right template.
                return template["generator"](self.generator._get_placeholders_for_template(template))
        return []


class AGIPipeline:
    def __init__(self):
        print("\nInitializing Declarative AGI Pipeline...")
        self.command_network = MockDeclarativeCommandNetwork()
        self.controller = LogicalController()
        print("AGI Pipeline Initialized.\n")

    def run_text_query(self, text: str):
        print(f"--- Running query for: '{text}' ---\n")

        # The real Encoder would be used here to feed the real CommandNetwork
        # vector = self.encoder.encode(text)

        declarations = self.command_network.process(text)

        print(f"\nCOMMAND_NETWORK produced {len(declarations)} declarations:")
        print(json.dumps(declarations, indent=2, ensure_ascii=False))

        self.controller.process_declarations(declarations)
        self.controller.print_graph_summary()


# Redefining the mock to be simpler and more direct for the final test
class FinalMockCommandNetwork:
    def __init__(self):
        self.generator = CommandNetworkDataGenerator()
        print("NEURAL_NETWORKS: Initialized FinalMockCommandNetwork.")

    def process(self, text: str) -> list:
        print(f"COMMAND_NETWORK: Simulating generation for text: '{text}'")
        text_lower = text.lower()
        gen = self.generator
        if "было" in text_lower and "добавили" in text_lower:
            return gen._generate_update_object_declarations({'<ENTITY>': 'слон', '<COUNT_WORD>': 'три', '<COUNT_WORD>_1': 'два'})
        elif "делает" in text_lower and "чтобы" in text_lower:
            return gen._generate_intent_dependency_declarations({'<PERSON>': 'начальник', '<TASK>': 'отчет', '<PERSON>_1': 'коллега', '<TASK>_1': 'презентация'})
        else:
            return gen._generate_create_object_declarations({'<LOCATION>': 'комната', '<COUNT_WORD>': 'два', '<ENTITY>': 'стул'})


if __name__ == '__main__':
    # We create a new pipeline instance for each run to have a clean graph

    print("\n\n=== Test Case 1: Simple Creation ===")
    pipeline1 = AGIPipeline()
    pipeline1.command_network = FinalMockCommandNetwork() # Use the final mock
    pipeline1.run_text_query("в комнате есть два стула")

    print("\n\n=== Test Case 2: Update ===")
    pipeline2 = AGIPipeline()
    pipeline2.command_network = FinalMockCommandNetwork()
    pipeline2.run_text_query("у нас было три слон, потом добавили еще два")

    print("\n\n=== Test Case 3: Intent Dependency ===")
    pipeline3 = AGIPipeline()
    pipeline3.command_network = FinalMockCommandNetwork()
    pipeline3.run_text_query("начальник делает отчет, чтобы коллега мог сделать презентация")
