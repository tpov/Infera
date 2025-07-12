from typing import Dict, Any, TYPE_CHECKING
import numpy as np

# This is a standard pattern to avoid circular imports while still allowing type hints.
# The `if TYPE_CHECKING:` block is only evaluated by static type checkers, not at runtime.
if TYPE_CHECKING:
    from src.logical_controller.controller import LogicalController

class MockReasoningNetwork:
    """
    A mock implementation of the Reasoning Network (e.g., a GNN).
    It does not perform any real graph analysis. Instead, it returns a
    pre-canned vector representing a plausible outcome based on simple
    checks of the nodes involved in a query.
    """
    def __init__(self, vector_size=768):
        """
        Initializes the mock reasoning network.
        Args:
            vector_size (int): The dimensionality of the response vector to generate.
        """
        self.vector_size = vector_size
        print("NEURAL_NETWORKS: Initialized MockReasoningNetwork.")

    def analyze(self, query_node_id: str, world_graph_controller: 'LogicalController') -> np.ndarray:
        """
        "Analyzes" the state of the world graph with respect to a query.

        Args:
            query_node_id (str): The ID of the node with state=QUERY that triggered the analysis.
            world_graph_controller (LogicalController): The controller instance holding the graph.

        Returns:
            np.ndarray: A mock vector representing the result of the reasoning process.
        """
        print(f"REASONING_NETWORK: Received query for node '{query_node_id}'. Analyzing graph.")

        query_node = world_graph_controller.get_node_by_id(query_node_id)
        if not query_node:
            print("REASONING_NETWORK: Query node not found!")
            return np.random.rand(self.vector_size) # Return random vector on error

        # --- Mock Logic ---
        # This simulates the GNN finding a rule like "tiger eats elephant".
        # We'll check if the query involves nodes named 'тигр' and 'слон'.

        # A real implementation would parse the formula `FUNC::PREDICT(id1, id2)`
        # For this mock, we'll just search the whole graph for relevant nodes.
        has_tiger = False
        has_elephant = False
        elephant_count = 0

        for node_id in world_graph_controller.world_graph.nodes:
            node = world_graph_controller.get_node_by_id(node_id)
            if node.name == 'тигр':
                has_tiger = True
            if node.name == 'слон':
                has_elephant = True
                elephant_count = node.attributes.get('count', 0)

        if has_tiger and has_elephant:
            print("REASONING_NETWORK: Found 'тигр' and 'слон'. Simulating 'tiger attacks elephant' outcome.")
            # We return a vector that the mock decoder will know how to interpret.
            # Let's create a "special" vector for this case.
            response_vector = np.ones(self.vector_size) * 0.5
            response_vector[0] = elephant_count # Embed the count in the vector
            return response_vector
        else:
            print("REASONING_NETWORK: No specific rule matched. Returning generic outcome.")
            return np.random.rand(self.vector_size)
