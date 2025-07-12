from typing import List, Dict, Any
import numpy as np

class MockCommandNetwork:
    """
    A mock implementation of the Command Network.
    It contains hardcoded rules to generate a sequence of commands based on keywords.
    This version is stateless to be more robust for testing.
    """
    def __init__(self):
        print("NEURAL_NETWORKS: Initialized MockCommandNetwork.")

    def process(self, text_input: str, vector_input: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        "Processes" a text input and returns a list of commands based on keyword matching.

        Args:
            text_input (str): The original text from the user.
            vector_input (np.ndarray, optional): The encoded vector. Ignored in this mock.

        Returns:
            List[Dict[str, Any]]: A list of command dictionaries.
        """
        print(f"COMMAND_NETWORK: Received text: '{text_input}'. Generating mock commands.")
        text_input = text_input.lower()
        commands = []

        # --- Rule for "слоны и тигр" ---
        if "слон" in text_input and "тигр" in text_input and "что будет" in text_input:
            elephant_node_id = "elephant_node_1"
            tiger_node_id = "tiger_node_1"
            query_node_id = "query_node_1"

            # 1. Create initial elephants
            commands.append({
                'action': 'CREATE_NODE',
                'node_type': 'Object',
                'data': {'name': 'слон', 'state': 'REAL', 'attributes': {'count': 2}, 'node_id': elephant_node_id}
            })
            # 2. Update them with a formula
            commands.append({
                'action': 'UPDATE_NODE_PROPERTY',
                'node_id': elephant_node_id,
                'property_key': 'count',
                'property_value': 'PREVIOUS_VALUE + 3'
            })
            # 3. Create the tiger
            commands.append({
                'action': 'CREATE_NODE',
                'node_type': 'Object',
                'data': {'name': 'тигр', 'state': 'HYPOTHETICAL', 'attributes': {'count': 1}, 'node_id': tiger_node_id}
            })
            # 4. Create an edge for context
            commands.append({
               'action': 'CREATE_EDGE',
               'source_id': tiger_node_id,
               'target_id': elephant_node_id,
               'edge_type': "IN_SAME_SCENARIO_AS"
            })
            # 5. Create the Query Node
            commands.append({
               'action': 'CREATE_NODE',
               'node_type': 'Object',
               'data': {
                   'name': 'world_state_query',
                   'state': 'QUERY',
                   'node_id': query_node_id,
                   'formula': {'outcome': f'FUNC::PREDICT({elephant_node_id}, {tiger_node_id})'}
               }
           })

        # --- Rule for just creating elephants ---
        elif "два слона" in text_input or "2 слона" in text_input:
            commands.append({
                'action': 'CREATE_NODE',
                'node_type': 'Object',
                'data': {'name': 'слон', 'state': 'REAL', 'attributes': {'count': 2}, 'node_id': 'elephant_node_2'}
            })

        if not commands:
            print("COMMAND_NETWORK: No mock rule matched.")

        return commands
