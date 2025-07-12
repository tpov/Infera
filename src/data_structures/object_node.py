import uuid
from typing import Dict, Any, Union, List
from enum import Enum

class NodeState(Enum):
    """
    Defines the existence state of a node.
    REAL: The node represents a confirmed, existing entity.
    HYPOTHETICAL: The node is part of a simulation or a "what-if" scenario.
    QUERY: The node represents a question about the world state, which needs to be resolved.
    """
    REAL = "REAL"
    HYPOTHETICAL = "HYPOTHETICAL"
    QUERY = "QUERY"

class ObjectNode:
    """
    Represents an Object in the World Graph.
    Objects are entities that have physical or conceptual embodiment.
    """
    def __init__(self,
                 name: str,
                 state: NodeState,
                 attributes: Dict[str, Any] = None,
                 time: Union[float, Dict[str, float]] = 0.0,
                 formula: Dict[str, str] = None,
                 node_id: str = None):
        """
        Initializes an ObjectNode.

        Args:
            name (str): The name or type of the object (e.g., 'elephant', 'tree').
            state (NodeState): The existence state of the node (REAL, HYPOTHETICAL, QUERY).
            attributes (Dict[str, Any], optional): A dictionary of properties like color, size, position, count.
                                                   The values can be direct or formulas. Defaults to {}.
            time (Union[float, Dict[str, float]], optional): A timestamp or a time interval. Defaults to 0.0.
            formula (Dict[str, str], optional): A dictionary where keys are attribute names and values are
                                                the formulas to compute them (e.g., {'count': 'PREVIOUS_VALUE + 3'}).
                                                Defaults to {}.
            node_id (str, optional): A unique identifier. If not provided, a new UUID will be generated.
        """
        self.id: str = node_id if node_id else str(uuid.uuid4())
        self.type: str = "Object"
        self.name: str = name
        self.state: NodeState = state
        self.time: Union[float, Dict[str, float]] = time

        # Attributes store the concrete values of the object's properties.
        self.attributes: Dict[str, Any] = attributes if attributes else {}

        # Formulas store the rules for how attributes can be calculated.
        # The LogicalController will parse these and update the attributes dict.
        self.formula: Dict[str, str] = formula if formula else {}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the node object to a dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "state": self.state.value,
            "time": self.time,
            "attributes": self.attributes,
            "formula": self.formula
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObjectNode':
        """Creates an ObjectNode instance from a dictionary."""
        return cls(
            name=data['name'],
            state=NodeState(data['state']),
            attributes=data.get('attributes', {}),
            time=data.get('time', 0.0),
            formula=data.get('formula', {}),
            node_id=data['id']
        )

    def __repr__(self) -> str:
        return f"ObjectNode(id={self.id}, name='{self.name}', state={self.state.value}, attributes={self.attributes})"
