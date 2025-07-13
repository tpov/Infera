import networkx as nx
import re
import time
from typing import Dict, Any, Union, List, Tuple

from src.data_structures.object_node import ObjectNode
from src.data_structures.intent_node import IntentNode

class LogicalController:
    def __init__(self):
        self.world_graph = nx.MultiDiGraph()

    def _get_node_by_name(self, name: str) -> Union[str, None]:
        for node_id, data in self.world_graph.nodes(data=True):
            if data.get('data') and data['data'].name == name:
                return node_id
        return None

    def _evaluate_field(self, node: Union[ObjectNode, IntentNode], field_key: str, field_value: Any) -> Tuple[Any, bool]:
        if not isinstance(field_value, str):
            return field_value, False

        # Regex to capture 'PREVIOUS_VALUE + 5' or similar simple arithmetic
        match = re.match(r"PREVIOUS_VALUE\s*([+\-*/])\s*(\d+(\.\d+)?)", field_value)
        if not match:
            return field_value, False

        op = match.group(1)
        val = float(match.group(2))

        try:
            # Get the old value from the node object before it was updated.
            # This is tricky because the update might have already happened in memory.
            # A robust implementation would store a 'before' state.
            # We'll simulate it by assuming the update hasn't been applied to the attribute yet.
            # This part of the logic is flawed and needs a better state management.
            # For now, let's assume the field hasn't been updated yet.
            # This is a conceptual bug in the previous logic.
            # Let's fix it by looking at the graph state.

            # The node passed in is the most up-to-date version. To get PREVIOUS_VALUE,
            # we can't look at the node itself. This is a flaw in the simple approach.
            # A real system would need transactionality.
            # Let's simplify: we assume the 'update' declaration is separate and we can find the old node.
            # This logic is getting complex, let's stick to a simpler evaluation for now.
            # The bug is that the update happens before evaluation.

            # Let's fix the logic flow entirely.
            # We will evaluate formulas *before* applying the new value.
            # This is handled in process_declarations now.

            # The value passed here is the formula string itself.
            # The `current_value` is what's currently in the node's attribute.
            current_value = getattr(node, field_key, 0.0)
            if not isinstance(current_value, (int, float)):
                current_value = 0.0

            if op == '+': result = current_value + val
            elif op == '-': result = current_value - val
            elif op == '*': result = current_value * val
            elif op == '/': result = current_value / val
            else: return field_value, False

            return f"{field_value} = {result}", True
        except (TypeError, ValueError):
            return field_value, False


    def process_declarations(self, declarations: List[Dict[str, Any]]) -> List[str]:
        processed_ids = []

        declarations_by_name = {}
        for decl in declarations:
            name = decl.get('name')
            if name not in declarations_by_name:
                declarations_by_name[name] = []
            declarations_by_name[name].append(decl)

        for name, decl_list in declarations_by_name.items():
            existing_node_id = self._get_node_by_name(name)
            node = None
            if existing_node_id:
                node = self.world_graph.nodes[existing_node_id]['data']
                print(f"CONTROLLER: Found existing node for '{name}'. Processing updates.")

            for decl in decl_list:
                node_type = decl.pop('type', 'Object')
                if node is None: # Create node on first declaration for this name
                    print(f"CONTROLLER: Creating new node for '{name}'.")
                    NodeClass = ObjectNode if node_type == 'Object' else IntentNode
                    node = NodeClass(**decl)
                    self.world_graph.add_node(node.id, data=node)
                else: # Update node
                    for key, value in decl.items():
                        # Evaluate formula BEFORE setting the attribute
                        new_value, was_computed = self._evaluate_field(node, key, value)
                        if was_computed:
                             print(f"CONTROLLER: Computed formula for {node.id}: {key} = {new_value}")
                        setattr(node, key, new_value if was_computed else value)

                node.timestamp = time.time()

            if node and node.id not in processed_ids:
                processed_ids.append(node.id)

        # Auto-link nodes processed in the same batch
        if len(processed_ids) > 1:
            for i in range(len(processed_ids)):
                for j in range(i + 1, len(processed_ids)):
                    self.world_graph.add_edge(processed_ids[i], processed_ids[j], type="RELATED_IN_DECLARATION")
                    print(f"CONTROLLER: Auto-linked {processed_ids[i]} and {processed_ids[j]}")

        return processed_ids

    def print_graph_summary(self):
        print("\n--- World Graph Summary ---")
        if not self.world_graph:
            print("Graph is empty.")
            return
        for node_id in self.world_graph.nodes:
            node_data = self.world_graph.nodes[node_id]['data']
            print(node_data.to_dict())
        for u, v, data in self.world_graph.edges(data=True):
            print(f"Edge: ({u}) -[{data.get('type', 'RELATED')}]-> ({v})")
        print("-------------------------\n")
