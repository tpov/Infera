import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
from enum import Enum

class StateCategory(Enum):
    EMOTIONAL = "emotional"
    PHYSICAL = "physical"
    LOGICAL = "logical"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    PROBABILISTIC = "probabilistic"
    INTENTIONAL = "intentional"
    RELATIONAL = "relational"

@dataclass
class CompressedState:
    """Сжатое описание состояния системы"""
    emotional_state: Dict[str, float] = field(default_factory=dict)  # эмоции, настроение
    physical_state: Dict[str, float] = field(default_factory=dict)   # физические параметры
    logical_state: Dict[str, float] = field(default_factory=dict)    # логические связи
    temporal_state: Dict[str, float] = field(default_factory=dict)   # временные параметры
    spatial_state: Dict[str, float] = field(default_factory=dict)    # пространственные параметры
    probabilistic_state: Dict[str, float] = field(default_factory=dict)  # вероятности
    intentional_state: Dict[str, float] = field(default_factory=dict)    # намерения
    relational_state: Dict[str, float] = field(default_factory=dict)     # отношения
    
    def to_vector(self) -> torch.Tensor:
        """Преобразует состояние в вектор"""
        all_states = []
        
        for category in StateCategory:
            state_dict = getattr(self, f"{category.value}_state")
            # Нормализуем значения
            values = list(state_dict.values())
            if values:
                values = np.array(values)
                values = (values - values.min()) / (values.max() - values.min() + 1e-8)
                all_states.extend(values)
            else:
                # Если нет значений, добавляем нули
                all_states.extend([0.0] * 10)  # 10 значений на категорию
        
        return torch.tensor(all_states, dtype=torch.float32)
    
    def to_text_description(self) -> str:
        """Преобразует состояние в текстовое описание"""
        descriptions = []
        
        for category in StateCategory:
            state_dict = getattr(self, f"{category.value}_state")
            if state_dict:
                # Находим топ-3 значения
                sorted_items = sorted(state_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                category_desc = f"{category.value}: " + ", ".join([f"{k}({v:.2f})" for k, v in sorted_items])
                descriptions.append(category_desc)
        
        return "; ".join(descriptions)

class StateCompressor(nn.Module):
    """Нейросеть для компрессии состояний системы"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, state_dim: int = 80):
        super(StateCompressor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Энкодер для входного вектора
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Слой для генерации состояний по категориям
        self.state_generators = nn.ModuleDict({
            category.value: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 10)  # 10 значений на категорию
            ) for category in StateCategory
        })
        
        # Attention механизм для фокусировки на важных аспектах
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, input_vector: torch.Tensor) -> CompressedState:
        """Сжимает входной вектор в состояние"""
        batch_size = input_vector.size(0)
        
        # Кодируем входной вектор
        encoded = self.encoder(input_vector)
        
        # Применяем attention
        encoded = encoded.unsqueeze(0)  # [1, batch_size, hidden_dim]
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = attended.squeeze(0)  # [batch_size, hidden_dim]
        
        # Генерируем состояния для каждой категории
        compressed_state = CompressedState()
        
        for category in StateCategory:
            generator = self.state_generators[category.value]
            state_values = generator(attended)  # [batch_size, 10]
            
            # Преобразуем в словарь
            state_dict = {}
            for i in range(10):
                state_dict[f"{category.value}_{i}"] = state_values[0, i].item()
            
            setattr(compressed_state, f"{category.value}_state", state_dict)
        
        return compressed_state

class StateDecompressor(nn.Module):
    """Нейросеть для декомпрессии состояний обратно в команды"""
    
    def __init__(self, state_dim: int = 80, hidden_dim: int = 256, output_dim: int = 768):
        super(StateDecompressor, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Декодер состояния
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # LSTM для генерации последовательности команд
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Выходной слой для команд
        self.command_output = nn.Linear(hidden_dim, 1000)  # Размер словаря команд
        
    def forward(self, state_vector: torch.Tensor, max_length: int = 50) -> torch.Tensor:
        """Декомпрессирует состояние в команды"""
        batch_size = state_vector.size(0)
        
        # Декодируем состояние
        decoded = self.decoder(state_vector)  # [batch_size, output_dim]
        
        # Подготавливаем для LSTM
        decoded = decoded.unsqueeze(1)  # [batch_size, 1, output_dim]
        
        # Инициализируем скрытое состояние
        h0 = torch.zeros(2, batch_size, self.hidden_dim).to(state_vector.device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim).to(state_vector.device)
        
        # Генерируем последовательность команд
        outputs = []
        current_input = decoded
        
        for _ in range(max_length):
            # LSTM шаг
            lstm_out, (h0, c0) = self.lstm(current_input, (h0, c0))
            
            # Выходной слой
            output = self.command_output(lstm_out)
            outputs.append(output)
            
            # Следующий вход
            current_input = output
        
        return torch.stack(outputs, dim=1)  # [batch_size, max_length, vocab_size]

class AdaptiveStateController:
    """Адаптивный контроллер состояний"""
    
    def __init__(self):
        self.compressor = StateCompressor()
        self.decompressor = StateDecompressor()
        self.current_state = CompressedState()
        self.state_history = []
        
    def update_state(self, input_vector: torch.Tensor, user_query: str) -> CompressedState:
        """Обновляет состояние на основе входного вектора и запроса"""
        
        # Сжимаем входной вектор в состояние
        new_state = self.compressor(input_vector)
        
        # Адаптируем состояние на основе запроса пользователя
        adapted_state = self._adapt_state_to_query(new_state, user_query)
        
        # Обновляем текущее состояние
        self.current_state = adapted_state
        self.state_history.append(adapted_state)
        
        return adapted_state
    
    def _adapt_state_to_query(self, state: CompressedState, query: str) -> CompressedState:
        """Адаптирует состояние под конкретный запрос"""
        query_lower = query.lower()
        
        # Адаптируем эмоциональное состояние
        if any(word in query_lower for word in ['грустно', 'печально', 'плохо']):
            state.emotional_state['sadness'] = 0.8
            state.emotional_state['happiness'] = 0.2
        
        if any(word in query_lower for word in ['радостно', 'хорошо', 'отлично']):
            state.emotional_state['happiness'] = 0.8
            state.emotional_state['sadness'] = 0.2
        
        # Адаптируем логическое состояние
        if any(word in query_lower for word in ['логично', 'правильно', 'верно']):
            state.logical_state['coherence'] = 0.9
            state.logical_state['contradiction'] = 0.1
        
        # Адаптируем вероятностное состояние
        if any(word in query_lower for word in ['возможно', 'вероятно', 'может']):
            state.probabilistic_state['uncertainty'] = 0.7
            state.probabilistic_state['certainty'] = 0.3
        
        return state
    
    def generate_commands_from_state(self, state: CompressedState) -> str:
        """Генерирует команды из состояния"""
        state_vector = state.to_vector()
        
        # Декомпрессируем состояние в команды
        with torch.no_grad():
            command_sequence = self.decompressor(state_vector.unsqueeze(0))
        
        # Преобразуем в текст команд (упрощенно)
        commands = self._vector_to_commands(command_sequence[0])
        return commands
    
    def _vector_to_commands(self, command_vector: torch.Tensor) -> str:
        """Преобразует вектор команд в текст"""
        # Упрощенная реализация - в реальности нужен словарь токенов
        commands = []
        
        # Анализируем состояние и генерируем соответствующие команды
        if self.current_state.emotional_state.get('happiness', 0) > 0.5:
            commands.append("set mood value positive")
        
        if self.current_state.logical_state.get('coherence', 0) > 0.7:
            commands.append("set logic_mode value coherent")
        
        if self.current_state.probabilistic_state.get('uncertainty', 0) > 0.5:
            commands.append("create uncertainty_handler quantity 1")
        
        return f"[{', '.join(commands)}]" if commands else ""
    
    def get_state_description(self) -> str:
        """Возвращает текстовое описание текущего состояния"""
        return self.current_state.to_text_description()

def create_state_training_data() -> List[Dict[str, Any]]:
    """Создает данные для обучения системы состояний"""
    training_data = []
    
    # Примеры различных состояний
    states = [
        {
            "query": "Я чувствую себя хорошо",
            "emotional_state": {"happiness": 0.8, "sadness": 0.1, "excitement": 0.6},
            "physical_state": {"energy": 0.7, "health": 0.9},
            "logical_state": {"coherence": 0.8, "clarity": 0.9}
        },
        {
            "query": "Это логично и правильно",
            "logical_state": {"coherence": 0.9, "validity": 0.8, "consistency": 0.7},
            "probabilistic_state": {"certainty": 0.8, "confidence": 0.9}
        },
        {
            "query": "Возможно, это так",
            "probabilistic_state": {"uncertainty": 0.7, "possibility": 0.8, "doubt": 0.4},
            "logical_state": {"tentative": 0.6, "exploration": 0.7}
        }
    ]
    
    for state in states:
        training_data.append({
            "input_vector": torch.randn(768),  # Заглушка
            "state": state,
            "expected_commands": f"set mood value {state.get('emotional_state', {}).get('happiness', 0.5)}"
        })
    
    return training_data