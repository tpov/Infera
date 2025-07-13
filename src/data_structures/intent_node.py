import uuid
from typing import Dict, Any, Union, List
from enum import Enum
# Import the updated context enum
from .object_node import NodeContext

class IntentStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"

class IntentNode:
    """
    Represents an Intent/Action in the World Graph.
    Context is now REAL or HYPOTHETICAL.
    """
    def __init__(self, **kwargs):
        # --- Core Identification ---
        self.id: str = kwargs.get('id', str(uuid.uuid4()))
        self.type: str = "Intent"
        self.name: str = kwargs.get('name', 'unknown_intent')
        self.owner_id: str = kwargs.get('owner_id')

        # --- Spatio-Temporal and Contextual ---
        self.timestamp: float = kwargs.get('timestamp')
        self.duration: float = kwargs.get('duration')
        self.context: NodeContext = NodeContext(kwargs.get('context', 'REAL'))

        # --- Query Flag ---
        self.is_query: bool = kwargs.get('is_query', False)

        # --- Action and Goal Parameters ---
        self.source_id: str = kwargs.get('source_id')
        self.target_id: str = kwargs.get('target_id')
        self.goal: str = kwargs.get('goal')
        self.value: Union[float, str] = kwargs.get('value')

        # --- Management and Execution ---
        self.priority: Union[int, str] = kwargs.get('priority', 5)
        self.status: IntentStatus = IntentStatus(kwargs.get('status', 'PENDING'))
        self.dependencies: List[str] = kwargs.get('dependencies', [])

        # --- Abstract Properties ---
        self.description: str = kwargs.get('description')
        self.probability: Union[float, str] = kwargs.get('probability', 1.0)

    def to_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
        return data

    def __repr__(self) -> str:
        return f"IntentNode(id={self.id}, name='{self.name}', status={self.status.value}, query={self.is_query})"
