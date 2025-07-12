# When running with `python -m src.main` from the project root,
# Python treats the project root as the starting point for imports.
# Therefore, all imports must be absolute from the `src` package.

from src.logical_controller.controller import LogicalController, NodeState
from src.neural_networks.encoder import MockEncoder
from src.neural_networks.command_network import MockCommandNetwork
from src.neural_networks.reasoning_network import MockReasoningNetwork
from src.neural_networks.decoder import MockDecoder

class AGIPipeline:
    """
    Orchestrates the entire AGI pipeline, connecting all the modules.
    """
    def __init__(self):
        print("Initializing AGI Pipeline...")
        self.encoder = MockEncoder()
        self.command_network = MockCommandNetwork()
        self.reasoning_network = MockReasoningNetwork()
        self.decoder = MockDecoder()
        self.controller = LogicalController()
        print("AGI Pipeline Initialized.\n")

    def run_text_query(self, text: str):
        """
        Runs a single text query through the entire pipeline.
        """
        print(f"--- Running query for: '{text}' ---\n")

        # 1. Encoder: Text -> Vector
        encoded_vector = self.encoder.encode(text)

        # 2. Command Network: Vector -> Commands
        commands = self.command_network.process(text, encoded_vector)
        print(f"\nCOMMAND_NETWORK generated {len(commands)} commands.")

        # 3. Logical Controller: Execute all commands first
        for cmd in commands:
            self.controller.execute_command(cmd)

        self.controller.print_graph_summary()

        # 4. AFTER all commands, check if a query needs to be resolved.
        # This is more robust.
        query_node_id_to_resolve = None
        for node_id, data in self.controller.world_graph.nodes(data=True):
            if data['data'].state == NodeState.QUERY:
                 query_node_id_to_resolve = node_id
                 # We take the first one we find. A more complex system might handle multiple queries.
                 break

        if query_node_id_to_resolve:
            print(f"--- Initiating Reasoning Cycle for Query Node: {query_node_id_to_resolve} ---\n")
            # 4a. Reasoning Network: (Graph + Query) -> Response Vector
            response_vector = self.reasoning_network.analyze(query_node_id_to_resolve, self.controller)

            # 4b. Decoder: Response Vector -> Text
            final_response = self.decoder.decode(response_vector)

            # 4c. Update the query node with the result
            query_node = self.controller.get_node_by_id(query_node_id_to_resolve)
            if query_node:
                # We also change its state to show it has been resolved.
                query_node.state = NodeState.REAL
                query_node.attributes['outcome_text'] = final_response
                print(f"CONTROLLER: Stored final response and resolved query node {query_node_id_to_resolve}.")

            print("\n--- Final Response ---")
            print(final_response)
            print("----------------------\n")
        else:
            print("\n--- No Query Detected ---")
            print("Commands executed, graph updated. No response generated.")
            print("-------------------------\n")


if __name__ == '__main__':
    # To run this script, stand in the root directory of the project (the one containing `src`)
    # and execute: python -m src.main
    pipeline = AGIPipeline()

    # --- Test Case 1: The full "elephants and tiger" query ---
    test_sentence = "К двум слонам добавили еще трех, а потом тигра. Что будет?"
    pipeline.run_text_query(test_sentence)

    # --- Test Case 2: A simple creation query without a question ---
    print("\n\n=== Starting New Test Case: Simple Update ===\n")
    pipeline_2 = AGIPipeline()
    test_sentence_2 = "Есть два слона."
    pipeline_2.run_text_query(test_sentence_2)
