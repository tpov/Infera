import uuid
from typing import Dict, Any, Union, List
from enum import Enum

class NodeContext(Enum):
    """Defines the reality context of a node."""
    REAL = "REAL"
    HYPOTHETICAL = "HYPOTHETICAL"

class ObjectDomain(Enum):
    """Defines the domain of an object, distinguishing physical from non-physical."""
    PHYSICAL = "PHYSICAL"
    VIRTUAL = "VIRTUAL"
    CONCEPTUAL = "CONCEPTUAL"

class PhysicalState(Enum):
    SOLID = "SOLID"
    LIQUID = "LIQUID"
    GAS = "GAS"
    UNKNOWN = "UNKNOWN"

class ObjectNode:
    """
    Represents a universal Object in the World Graph.
    Can be physical, virtual, or conceptual.
    """
    def __init__(self, **kwargs):
        # --- Core Identification ---
        self.id: str = kwargs.get('id', str(uuid.uuid4()))
        self.type: str = "Object"
        self.name: str = kwargs.get('name', 'unknown_object')
        self.owner_id: str = kwargs.get('owner_id')
        self.domain: ObjectDomain = ObjectDomain(kwargs.get('domain', 'PHYSICAL'))

        # --- Spatio-Temporal and Contextual ---
        self.timestamp: float = kwargs.get('timestamp')
        self.duration: float = kwargs.get('duration')
        self.context: NodeContext = NodeContext(kwargs.get('context', 'REAL'))
        self.position: Dict[str, float] = kwargs.get('position')

        # --- Query Flag ---
        self.is_query: bool = kwargs.get('is_query', False)

        # --- Quantitative and Qualitative Attributes ---
        self.count: Union[int, str] = kwargs.get('count', 1)
        self.size: Dict[str, float] = kwargs.get('size')
        self.weight: Union[float, str] = kwargs.get('weight')
        self.color: str = kwargs.get('color')
        self.physical_state: PhysicalState = PhysicalState(kwargs.get('physical_state', 'UNKNOWN'))

        # --- Abstract Properties ---
        self.description: str = kwargs.get('description')
        self.probability: Union[float, str] = kwargs.get('probability', 1.0)
        self.integrity: Union[float, str] = kwargs.get('integrity', 1.0)

    def to_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
        return data

    def __repr__(self) -> str:
        return f"ObjectNode(id={self.id}, name='{self.name}', domain={self.domain.value}, context={self.context.value}, query={self.is_query})"
