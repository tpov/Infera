import uuid
from typing import Dict, Any, Union
from enum import Enum
# Use a relative import within the same package
from .object_node import NodeState

class IntentType(Enum):
    """
    Defines the specific type of an Intent.
    """
    DESIRE = "DESIRE"
    OPINION = "OPINION"
    GOAL = "GOAL"
    ACTION = "ACTION"
    FEELING = "FEELING"
    QUESTION = "QUESTION" # Explicitly for "what will happen?" type questions

class IntentNode:
    """
    Represents an Intent in the World Graph.
    Intents are non-physical concepts like desires, goals, or actions.
    """
    def __init__(self,
                 intent_type: IntentType,
                 state: NodeState,
                 parameters: Dict[str, Any] = None,
                 time: Union[float, Dict[str, float]] = 0.0,
                 formula: Dict[str, str] = None,
                 node_id: str = None):
        """
        Initializes an IntentNode.

        Args:
            intent_type (IntentType): The specific type of the intent (e.g., ACTION, GOAL).
            state (NodeState): The existence state of the node (REAL, HYPOTHETICAL, QUERY).
            parameters (Dict[str, Any], optional): A dictionary of parameters for the intent,
                                                   such as source_id, target_id, value. Defaults to {}.
            time (Union[float, Dict[str, float]], optional): A timestamp or a time interval. Defaults to 0.0.
            formula (Dict[str, str], optional): Formulas to compute parameter values. Defaults to {}.
            node_id (str, optional): A unique identifier. If not provided, a new UUID will be generated.
        """
        self.id: str = node_id if node_id else str(uuid.uuid4())
        self.type: str = "Intent"
        self.intent_type: IntentType = intent_type
        self.state: NodeState = state
        self.time: Union[float, Dict[str, float]] = time

        # Parameters store the concrete values for the intent's execution.
        self.parameters: Dict[str, Any] = parameters if parameters else {}

        # Formulas store rules for calculating parameters.
        self.formula: Dict[str, str] = formula if formula else {}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the node object to a dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "intent_type": self.intent_type.value,
            "state": self.state.value,
            "time": self.time,
            "parameters": self.parameters,
            "formula": self.formula
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntentNode':
        """Creates an IntentNode instance from a dictionary."""
        return cls(
            intent_type=IntentType(data['intent_type']),
            state=NodeState(data['state']),
            parameters=data.get('parameters', {}),
            time=data.get('time', 0.0),
            formula=data.get('formula', {}),
            node_id=data['id']
        )

    def __repr__(self) -> str:
        return f"IntentNode(id={self.id}, type='{self.intent_type.value}', state={self.state.value}, parameters={self.parameters})"
