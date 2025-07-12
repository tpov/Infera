import networkx as nx
import re
from typing import Dict, Any, Union

from src.data_structures.object_node import ObjectNode, NodeState
from src.data_structures.intent_node import IntentNode, IntentType


class LogicalController:
    """
    The deterministic core of the AGI. It manages the World Graph and executes commands.
    It is NOT a neural network. It is a state machine that operates on the graph.
    """
    def __init__(self):
        # For simplicity, we use networkx for the in-memory graph.
        # This can be replaced with a more robust graph database like Neo4j later.
        self.world_graph = nx.MultiDiGraph()
        # A simple regex to find formulas like 'PREVIOUS_VALUE + 3'
        self.formula_pattern = re.compile(r"PREVIOUS_VALUE\s*([+\-*/])\s*(\d+)")
        # A simple regex for function-like formulas
        self.func_pattern = re.compile(r"FUNC::(\w+)\((.*)\)")

    def _resolve_node(self, node_id: str) -> Union[ObjectNode, IntentNode, None]:
        """Gets the node data from the graph."""
        if self.world_graph.has_node(node_id):
            return self.world_graph.nodes[node_id]['data']
        return None

    def _parse_and_compute_formula(self, formula: str, target_node: Union[ObjectNode, IntentNode], attr_key: str) -> Any:
        """
        Parses and computes the value of a formula string.
        Phase 1: Handles simple 'PREVIOUS_VALUE' arithmetic.
        """
        match = self.formula_pattern.match(formula)
        if match:
            operator = match.group(1)
            value = float(match.group(2))

            # Get the previous value from the node's attributes or parameters
            if isinstance(target_node, ObjectNode):
                previous_value = target_node.attributes.get(attr_key, 0.0)
            elif isinstance(target_node, IntentNode):
                previous_value = target_node.parameters.get(attr_key, 0.0)
            else:
                previous_value = 0.0

            if operator == '+':
                return previous_value + value
            elif operator == '-':
                return previous_value - value
            elif operator == '*':
                return previous_value * value
            elif operator == '/':
                return previous_value / value if value != 0 else float('inf')

        # Placeholder for function calls like FUNC::PREDICT(id1, id2)
        func_match = self.func_pattern.match(formula)
        if func_match:
            # For now, we just return the string indicating a function call is needed
            # The main loop will need to see this and trigger the ReasoningNetwork
            return f"PENDING_FUNC::{func_match.group(1)}({func_match.group(2)})"

        # If no pattern matches, return the formula string as is
        return formula


    def execute_command(self, command: Dict[str, Any]):
        """
        Executes a single command from the Command Network.

        Args:
            command (Dict[str, Any]): A dictionary representing the command.
                Example:
                {
                    'action': 'CREATE_NODE',
                    'node_type': 'Object',
                    'data': {'name': 'elephant', 'state': 'REAL', 'attributes': {'count': 2}}
                }
                {
                    'action': 'UPDATE_NODE_PROPERTY',
                    'node_id': 'some-uuid',
                    'property_key': 'count',
                    'property_value': 'PREVIOUS_VALUE + 3'
                }
        """
        action = command.get("action")

        if action == "CREATE_NODE":
            node_type = command.get("node_type")
            data = command.get("data", {})
            node = None
            if node_type == "Object":
                data['state'] = NodeState(data.get('state', 'REAL'))
                node = ObjectNode(**data)
            elif node_type == "Intent":
                data['state'] = NodeState(data.get('state', 'REAL'))
                data['intent_type'] = IntentType(data.get('intent_type', 'ACTION'))
                node = IntentNode(**data)

            if node:
                self.world_graph.add_node(node.id, data=node)
                print(f"CONTROLLER: Created Node {node}")
                # Check for formulas upon creation
                self._process_formulas(node)


        elif action == "UPDATE_NODE_PROPERTY":
            node_id = command.get("node_id")
            key = command.get("property_key")
            value = command.get("property_value") # This could be a direct value or a formula string
            node = self._resolve_node(node_id)
            if node and key:
                # We assume the value is a formula and try to compute it.
                # If it's not a formula, _parse_and_compute_formula will return it as is.
                computed_value = self._parse_and_compute_formula(str(value), node, key)

                if isinstance(node, ObjectNode):
                    node.attributes[key] = computed_value
                elif isinstance(node, IntentNode):
                    node.parameters[key] = computed_value
                print(f"CONTROLLER: Updated Node {node_id}. Set {key} to {computed_value}")

        elif action == "CREATE_EDGE":
            source_id = command.get("source_id")
            target_id = command.get("target_id")
            edge_type = command.get("edge_type", "RELATED_TO")
            if self.world_graph.has_node(source_id) and self.world_graph.has_node(target_id):
                self.world_graph.add_edge(source_id, target_id, key=edge_type, type=edge_type)
                print(f"CONTROLLER: Created Edge from {source_id} to {target_id} of type {edge_type}")


    def _process_formulas(self, node: Union[ObjectNode, IntentNode]):
        """
        Iterates through a node's formulas and computes them, updating the node's attributes/parameters.
        """
        # We need to copy the items because the dictionary might be modified during iteration
        # if one formula depends on another (not handled yet, but good practice).
        formulas_to_process = list(node.formula.items())

        for key, formula_str in formulas_to_process:
            computed_value = self._parse_and_compute_formula(formula_str, node, key)
            if isinstance(node, ObjectNode):
                node.attributes[key] = computed_value
                print(f"CONTROLLER: Computed formula for {node.id}: {key} = {computed_value}")
            elif isinstance(node, IntentNode):
                node.parameters[key] = computed_value
                print(f"CONTROLLER: Computed formula for {node.id}: {key} = {computed_value}")

    def get_node_by_id(self, node_id: str):
        return self._resolve_node(node_id)

    def print_graph_summary(self):
        print("\n--- World Graph Summary ---")
        if not self.world_graph.nodes:
            print("Graph is empty.")
            return
        for node_id in self.world_graph.nodes:
            node_data = self.world_graph.nodes[node_id]['data']
            print(node_data)
        for edge in self.world_graph.edges(keys=True):
            print(f"Edge: ({edge[0]}) -[{edge[2]}]-> ({edge[1]})")
        print("-------------------------\n")
