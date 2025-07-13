import uuid
from typing import Dict, Any, Union, List
from enum import Enum
from .object_node import NodeState # Reuse NodeState

class IntentStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"

class IntentNode:
    """
    Represents an Intent/Action in the World Graph. A declarative data structure.
    All fields are optional and can be formulas (represented as strings).
    """
    def __init__(self, **kwargs):
        # --- Core Identification ---
        self.id: str = kwargs.get('id', str(uuid.uuid4()))
        self.type: str = "Intent"
        self.name: str = kwargs.get('name', 'unknown_intent') # e.g., 'task_to_buy_milk'
        self.owner_id: str = kwargs.get('owner_id')

        # --- Spatio-Temporal and Contextual ---
        self.timestamp: float = kwargs.get('timestamp')
        self.duration: float = kwargs.get('duration')
        self.context: NodeState = NodeState(kwargs.get('context', 'REAL'))

        # --- Action and Goal Parameters ---
        self.source_id: str = kwargs.get('source_id') # Who/what initiates the intent
        self.target_id: str = kwargs.get('target_id') # Who/what is the target of the intent
        self.goal: str = kwargs.get('goal') # Text description of the desired outcome
        self.value: Union[float, str] = kwargs.get('value') # e.g., money, score, etc.

        # --- Management and Execution ---
        self.priority: Union[int, str] = kwargs.get('priority', 5)
        self.status: IntentStatus = IntentStatus(kwargs.get('status', 'PENDING'))
        self.dependencies: List[str] = kwargs.get('dependencies', []) # List of other Intent IDs

        # --- Abstract Properties ---
        self.description: str = kwargs.get('description')
        self.probability: Union[float, str] = kwargs.get('probability', 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the node to a dictionary, handling enums."""
        data = self.__dict__.copy()
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
        return data

    def __repr__(self) -> str:
        return f"IntentNode(id={self.id}, name='{self.name}', status={self.status.value})"
