import uuid
from typing import Dict, Any, Union, List
from enum import Enum

class NodeState(Enum):
    REAL = "REAL"
    HYPOTHETICAL = "HYPOTHETICAL"
    QUERY = "QUERY"

class PhysicalState(Enum):
    SOLID = "SOLID"
    LIQUID = "LIQUID"
    GAS = "GAS"
    UNKNOWN = "UNKNOWN"

class ObjectNode:
    """
    Represents an Object in the World Graph. A declarative data structure.
    All fields are optional and can be formulas (represented as strings).
    """
    def __init__(self, **kwargs):
        # --- Core Identification ---
        self.id: str = kwargs.get('id', str(uuid.uuid4()))
        self.type: str = "Object"
        self.name: str = kwargs.get('name', 'unknown_object')
        self.owner_id: str = kwargs.get('owner_id')

        # --- Spatio-Temporal and Contextual ---
        self.timestamp: float = kwargs.get('timestamp')
        self.duration: float = kwargs.get('duration')
        self.context: NodeState = NodeState(kwargs.get('context', 'REAL'))
        self.position: Dict[str, float] = kwargs.get('position') # {'x': 0, 'y': 0, 'z': 0}

        # --- Quantitative and Qualitative Attributes ---
        self.count: Union[int, str] = kwargs.get('count', 1)
        self.size: Dict[str, float] = kwargs.get('size') # {'length': 0, 'width': 0, 'height': 0}
        self.weight: Union[float, str] = kwargs.get('weight')
        self.color: str = kwargs.get('color')
        self.physical_state: PhysicalState = PhysicalState(kwargs.get('physical_state', 'UNKNOWN'))

        # --- Abstract Properties ---
        self.description: str = kwargs.get('description')
        self.probability: Union[float, str] = kwargs.get('probability', 1.0)
        self.integrity: Union[float, str] = kwargs.get('integrity', 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the node to a dictionary, handling enums."""
        data = self.__dict__.copy()
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
        return data

    def __repr__(self) -> str:
        return f"ObjectNode(id={self.id}, name='{self.name}', context={self.context.value})"
