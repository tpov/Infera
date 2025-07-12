import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import random

# ============================================================================
# ENHANCED STATE MANAGEMENT
# ============================================================================

class MentalState(Enum):
    """Психические состояния системы"""
    CURIOUS = "curious"
    CONFUSED = "confused"
    CONFIDENT = "confident"
    ANALYZING = "analyzing"
    CREATIVE = "creative"
    LOGICAL = "logical"
    EMOTIONAL = "emotional"
    NEUTRAL = "neutral"

class ContextType(Enum):
    """Типы контекста"""
    QUESTION = "question"
    STATEMENT = "statement"
    COMMAND = "command"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"
    ASSOCIATION = "association"

@dataclass
class ContextNode:
    """Узел контекста с семантическими связями"""
    id: str
    content: str
    context_type: ContextType
    confidence: float = 0.5
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    connections: Set[str] = field(default_factory=set)  # ID связанных узлов
    semantic_vector: Optional[List[float]] = None
    emotional_tone: float = 0.0  # -1 (негатив) до 1 (позитив)
    logical_complexity: float = 0.5
    hidden_meaning: Optional[str] = None

@dataclass
class SystemState:
    """Расширенное состояние системы"""
    # Базовые объекты
    objects: Dict[str, Any] = field(default_factory=dict)
    
    # Психическое состояние
    mental_state: MentalState = MentalState.NEUTRAL
    emotional_state: float = 0.0  # -1 до 1
    confidence_level: float = 0.5
    curiosity_level: float = 0.5
    creativity_level: float = 0.5
    
    # Контекстная память
    context_graph: Dict[str, ContextNode] = field(default_factory=dict)
    active_contexts: Set[str] = field(default_factory=set)
    context_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Логические цепочки
    logical_chains: List[List[str]] = field(default_factory=list)
    causal_relationships: Dict[str, List[str]] = field(default_factory=dict)
    hidden_patterns: List[str] = field(default_factory=list)
    
    # Семантические ассоциации
    semantic_network: Dict[str, Set[str]] = field(default_factory=dict)
    concept_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    
    # Временные метки
    last_update: datetime = field(default_factory=datetime.now)
    session_start: datetime = field(default_factory=datetime.now)

# ============================================================================
# CONTEXTUAL COMMAND GENERATOR
# ============================================================================

class ContextualCommandGenerator(nn.Module):
    """Нейросеть для генерации команд с учетом контекста"""
    
    def __init__(self, input_dim=768, hidden_dim=512, context_dim=256):
        super(ContextualCommandGenerator, self).__init__()
        
        # Энкодер для входного текста
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Энкодер для контекста
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Объединяющий слой
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM для генерации последовательности команд
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Выходные слои
        self.command_head = nn.Linear(hidden_dim, 1000)  # Словарь команд
        self.context_head = nn.Linear(hidden_dim, context_dim)  # Новый контекст
        
    def forward(self, text_vector, context_vector=None, max_length=20):
        batch_size = text_vector.size(0)
        
        # Кодируем текст
        text_encoded = self.text_encoder(text_vector)
        
        # Кодируем контекст (если есть)
        if context_vector is not None:
            context_encoded = self.context_encoder(context_vector)
            combined = torch.cat([text_encoded, context_encoded], dim=1)
        else:
            # Если контекста нет, используем только текст
            combined = text_encoded
        
        # Объединяем
        combined = self.combiner(combined)
        
        # Подготавливаем для LSTM
        combined = combined.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Инициализируем LSTM
        h0 = torch.zeros(2, batch_size, combined.size(-1)).to(text_vector.device)
        c0 = torch.zeros(2, batch_size, combined.size(-1)).to(text_vector.device)
        
        # Генерируем команды
        commands = []
        contexts = []
        current_input = combined
        
        for _ in range(max_length):
            lstm_out, (h0, c0) = self.lstm(current_input, (h0, c0))
            
            # Генерируем команду
            command_logits = self.command_head(lstm_out)
            commands.append(command_logits)
            
            # Генерируем новый контекст
            context_output = self.context_head(lstm_out)
            contexts.append(context_output)
            
            # Следующий вход
            current_input = command_logits
        
        return torch.stack(commands, dim=1), torch.stack(contexts, dim=1)

# ============================================================================
# STATE-AWARE CONTROLLER
# ============================================================================

class StateAwareController:
    """Контроллер, осведомленный о состоянии системы"""
    
    def __init__(self):
        self.state = SystemState()
        self.command_generator = ContextualCommandGenerator()
        self.context_analyzer = ContextAnalyzer()
        self.logic_engine = LogicEngine()
        
    def process_input(self, user_input: str, input_vector: List[float]) -> Dict[str, Any]:
        """Обрабатывает вход пользователя с учетом состояния"""
        
        # 1. Анализируем контекст
        context_analysis = self.context_analyzer.analyze(user_input, self.state)
        
        # 2. Обновляем состояние
        self._update_mental_state(context_analysis)
        
        # 3. Генерируем команды с учетом контекста
        commands = self._generate_contextual_commands(input_vector, context_analysis)
        
        # 4. Выполняем команды
        execution_result = self._execute_commands(commands)
        
        # 5. Обновляем логические цепочки
        self._update_logical_chains(user_input, execution_result)
        
        # 6. Генерируем ответ
        response = self._generate_response(context_analysis, execution_result)
        
        return {
            "response": response,
            "mental_state": self.state.mental_state.value,
            "confidence": self.state.confidence_level,
            "context_analysis": context_analysis,
            "execution_result": execution_result,
            "logical_chains": self.state.logical_chains[-5:],  # Последние 5 цепочек
            "hidden_patterns": self.state.hidden_patterns[-3:]  # Последние 3 паттерна
        }
    
    def _update_mental_state(self, context_analysis: Dict[str, Any]):
        """Обновляет психическое состояние на основе анализа контекста"""
        
        # Анализируем эмоциональный тон
        emotional_tone = context_analysis.get('emotional_tone', 0.0)
        self.state.emotional_state = (self.state.emotional_state + emotional_tone) / 2
        
        # Обновляем уровень уверенности
        if context_analysis.get('is_question', False):
            self.state.curiosity_level = min(1.0, self.state.curiosity_level + 0.1)
            self.state.confidence_level = max(0.1, self.state.confidence_level - 0.05)
        else:
            self.state.confidence_level = min(1.0, self.state.confidence_level + 0.05)
        
        # Определяем психическое состояние
        if self.state.curiosity_level > 0.7:
            self.state.mental_state = MentalState.CURIOUS
        elif self.state.confidence_level < 0.3:
            self.state.mental_state = MentalState.CONFUSED
        elif self.state.confidence_level > 0.8:
            self.state.mental_state = MentalState.CONFIDENT
        elif context_analysis.get('logical_complexity', 0) > 0.7:
            self.state.mental_state = MentalState.LOGICAL
        else:
            self.state.mental_state = MentalState.NEUTRAL
    
    def _generate_contextual_commands(self, input_vector: List[float], 
                                   context_analysis: Dict[str, Any]) -> List[str]:
        """Генерирует команды с учетом контекста"""
        
        # Преобразуем в тензоры
        text_tensor = torch.tensor(input_vector).unsqueeze(0).float()
        
        # Создаем контекстный вектор
        context_vector = self._create_context_vector(context_analysis)
        context_tensor = torch.tensor(context_vector).unsqueeze(0).float()
        
        # Генерируем команды
        with torch.no_grad():
            commands, new_contexts = self.command_generator(text_tensor, context_tensor)
        
        # Декодируем команды
        decoded_commands = self._decode_commands(commands[0])
        
        return decoded_commands
    
    def _create_context_vector(self, context_analysis: Dict[str, Any]) -> List[float]:
        """Создает вектор контекста"""
        vector = []
        
        # Эмоциональный тон
        vector.append(context_analysis.get('emotional_tone', 0.0))
        
        # Логическая сложность
        vector.append(context_analysis.get('logical_complexity', 0.5))
        
        # Тип контекста (one-hot encoding)
        context_types = ['question', 'statement', 'command', 'observation']
        for ctx_type in context_types:
            vector.append(1.0 if context_analysis.get('context_type') == ctx_type else 0.0)
        
        # Уверенность
        vector.append(self.state.confidence_level)
        
        # Любопытство
        vector.append(self.state.curiosity_level)
        
        # Дополняем до нужной размерности
        while len(vector) < 256:
            vector.append(0.0)
        
        return vector[:256]
    
    def _decode_commands(self, command_logits: torch.Tensor) -> List[str]:
        """Декодирует команды из логгитов"""
        # Простая реализация - выбираем топ-3 команды
        _, indices = torch.topk(command_logits, 3)
        
        # Базовый словарь команд
        command_vocab = [
            "analyze_context", "create_hypothesis", "find_patterns",
            "build_logical_chain", "extract_meaning", "generate_response",
            "update_knowledge", "create_association", "solve_problem",
            "explain_concept", "predict_outcome", "synthesize_info"
        ]
        
        commands = []
        for idx in indices:
            if idx < len(command_vocab):
                commands.append(command_vocab[idx])
        
        return commands
    
    def _execute_commands(self, commands: List[str]) -> Dict[str, Any]:
        """Выполняет команды"""
        results = {
            "executed_commands": [],
            "created_objects": [],
            "updated_contexts": [],
            "discovered_patterns": []
        }
        
        for command in commands:
            if command == "analyze_context":
                results["executed_commands"].append("analyze_context")
                # Анализируем текущий контекст
                pass
            elif command == "create_hypothesis":
                results["executed_commands"].append("create_hypothesis")
                # Создаем гипотезу
                pass
            elif command == "find_patterns":
                results["executed_commands"].append("find_patterns")
                # Ищем паттерны
                pass
        
        return results
    
    def _update_logical_chains(self, user_input: str, execution_result: Dict[str, Any]):
        """Обновляет логические цепочки"""
        # Добавляем новый узел в цепочку
        chain_id = f"chain_{len(self.state.logical_chains)}"
        
        # Создаем контекстный узел
        context_node = ContextNode(
            id=chain_id,
            content=user_input,
            context_type=ContextType.STATEMENT,
            confidence=self.state.confidence_level,
            importance=0.7
        )
        
        self.state.context_graph[chain_id] = context_node
        self.state.active_contexts.add(chain_id)
        
        # Добавляем в историю
        self.state.context_history.append(chain_id)
        
        # Обновляем логические цепочки
        if len(self.state.logical_chains) > 0:
            # Связываем с предыдущей цепочкой
            self.state.logical_chains[-1].append(chain_id)
        else:
            # Создаем новую цепочку
            self.state.logical_chains.append([chain_id])
    
    def _generate_response(self, context_analysis: Dict[str, Any], 
                         execution_result: Dict[str, Any]) -> str:
        """Генерирует ответ на основе контекста и результатов выполнения"""
        
        # Простая логика генерации ответа
        if context_analysis.get('is_question', False):
            if self.state.mental_state == MentalState.CURIOUS:
                return "Это интересный вопрос! Давайте разберем его подробнее..."
            elif self.state.mental_state == MentalState.CONFUSED:
                return "Хм, это сложный вопрос. Мне нужно подумать..."
            else:
                return "Позвольте мне проанализировать ваш вопрос..."
        else:
            if self.state.mental_state == MentalState.CONFIDENT:
                return "Понимаю! Это важная информация."
            else:
                return "Спасибо за информацию. Я это учту."
    
    def get_system_state_summary(self) -> Dict[str, Any]:
        """Возвращает краткое описание состояния системы"""
        return {
            "mental_state": self.state.mental_state.value,
            "emotional_state": self.state.emotional_state,
            "confidence_level": self.state.confidence_level,
            "curiosity_level": self.state.curiosity_level,
            "active_contexts": len(self.state.active_contexts),
            "logical_chains": len(self.state.logical_chains),
            "hidden_patterns": len(self.state.hidden_patterns),
            "context_history_length": len(self.state.context_history)
        }

# ============================================================================
# CONTEXT ANALYZER
# ============================================================================

class ContextAnalyzer:
    """Анализатор контекста для понимания скрытого смысла"""
    
    def __init__(self):
        self.emotional_keywords = {
            'positive': ['хорошо', 'отлично', 'прекрасно', 'радость', 'счастье'],
            'negative': ['плохо', 'ужасно', 'грусть', 'злость', 'разочарование'],
            'neutral': ['обычно', 'нормально', 'стандартно']
        }
        
        self.question_indicators = ['что', 'как', 'почему', 'когда', 'где', 'кто', '?']
        self.command_indicators = ['сделай', 'выполни', 'создай', 'удали', 'измени']
        
    def analyze(self, text: str, system_state: SystemState) -> Dict[str, Any]:
        """Анализирует контекст текста"""
        
        text_lower = text.lower()
        
        # Определяем тип контекста
        context_type = self._determine_context_type(text_lower)
        
        # Анализируем эмоциональный тон
        emotional_tone = self._analyze_emotional_tone(text_lower)
        
        # Определяем логическую сложность
        logical_complexity = self._analyze_logical_complexity(text)
        
        # Ищем скрытый смысл
        hidden_meaning = self._find_hidden_meaning(text, system_state)
        
        # Определяем, является ли это вопросом
        is_question = any(indicator in text_lower for indicator in self.question_indicators)
        
        return {
            'context_type': context_type,
            'emotional_tone': emotional_tone,
            'logical_complexity': logical_complexity,
            'hidden_meaning': hidden_meaning,
            'is_question': is_question,
            'text_length': len(text),
            'word_count': len(text.split())
        }
    
    def _determine_context_type(self, text: str) -> str:
        """Определяет тип контекста"""
        if any(indicator in text for indicator in self.question_indicators):
            return 'question'
        elif any(indicator in text for indicator in self.command_indicators):
            return 'command'
        elif any(word in text for word in ['вижу', 'наблюдаю', 'замечаю']):
            return 'observation'
        else:
            return 'statement'
    
    def _analyze_emotional_tone(self, text: str) -> float:
        """Анализирует эмоциональный тон"""
        positive_count = sum(1 for word in self.emotional_keywords['positive'] if word in text)
        negative_count = sum(1 for word in self.emotional_keywords['negative'] if word in text)
        
        total_emotional_words = positive_count + negative_count
        if total_emotional_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_emotional_words
    
    def _analyze_logical_complexity(self, text: str) -> float:
        """Анализирует логическую сложность"""
        # Простая эвристика: больше слов = сложнее
        words = text.split()
        complexity = min(1.0, len(words) / 20.0)  # Нормализуем до 1.0
        
        # Учитываем наличие логических связок
        logical_connectors = ['потому что', 'следовательно', 'поэтому', 'если', 'то', 'иначе']
        connector_count = sum(1 for connector in logical_connectors if connector in text)
        complexity += connector_count * 0.1
        
        return min(1.0, complexity)
    
    def _find_hidden_meaning(self, text: str, system_state: SystemState) -> Optional[str]:
        """Ищет скрытый смысл в тексте"""
        # Простая реализация - ищем ключевые слова
        hidden_indicators = ['на самом деле', 'по сути', 'в действительности', 'скрыто']
        
        for indicator in hidden_indicators:
            if indicator in text.lower():
                return f"Обнаружен скрытый смысл: {indicator}"
        
        return None

# ============================================================================
# LOGIC ENGINE
# ============================================================================

class LogicEngine:
    """Движок логики для построения цепочек рассуждений"""
    
    def __init__(self):
        self.causal_patterns = [
            (r'если (.+), то (.+)', 'conditional'),
            (r'(.+) приводит к (.+)', 'causal'),
            (r'(.+) потому что (.+)', 'explanation'),
            (r'(.+) следовательно (.+)', 'inference')
        ]
    
    def build_logical_chain(self, premises: List[str]) -> List[str]:
        """Строит логическую цепочку из посылок"""
        chain = []
        
        for premise in premises:
            # Анализируем посылку на наличие логических связей
            for pattern, logic_type in self.causal_patterns:
                match = re.search(pattern, premise, re.IGNORECASE)
                if match:
                    if logic_type == 'conditional':
                        chain.append(f"Условие: {match.group(1)}")
                        chain.append(f"Следствие: {match.group(2)}")
                    elif logic_type == 'causal':
                        chain.append(f"Причина: {match.group(1)}")
                        chain.append(f"Результат: {match.group(2)}")
        
        return chain
    
    def find_patterns(self, data: List[str]) -> List[str]:
        """Ищет паттерны в данных"""
        patterns = []
        
        # Простая реализация - ищем повторяющиеся элементы
        word_freq = defaultdict(int)
        for item in data:
            words = item.lower().split()
            for word in words:
                word_freq[word] += 1
        
        # Находим частые слова
        for word, freq in word_freq.items():
            if freq > 2:  # Слово встречается больше 2 раз
                patterns.append(f"Частое слово: {word} (встречается {freq} раз)")
        
        return patterns

# ============================================================================
# MAIN AGI SYSTEM
# ============================================================================

class EnhancedAGISystem:
    """Улучшенная AGI система с пониманием контекста"""
    
    def __init__(self):
        self.controller = StateAwareController()
        self.vectorizer = None  # Будет инициализирован позже
        
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Обрабатывает вход пользователя"""
        
        # Создаем простой вектор (заглушка)
        input_vector = [random.random() for _ in range(768)]
        
        # Обрабатываем через контроллер
        result = self.controller.process_input(user_input, input_vector)
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Возвращает статус системы"""
        return self.controller.get_system_state_summary()

# ============================================================================
# TESTING
# ============================================================================

def test_enhanced_agi():
    """Тестирует улучшенную AGI систему"""
    print("Тестирование улучшенной AGI системы...")
    
    system = EnhancedAGISystem()
    
    # Тестовые запросы
    test_inputs = [
        "Что такое искусственный интеллект?",
        "Мне грустно сегодня",
        "Если идет дождь, то улицы мокрые",
        "Создай систему автоматизации",
        "Почему небо голубое?"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- Тест {i} ---")
        print(f"Вход: {user_input}")
        
        result = system.process_input(user_input)
        
        print(f"Ответ: {result['response']}")
        print(f"Психическое состояние: {result['mental_state']}")
        print(f"Уверенность: {result['confidence']:.2f}")
        print(f"Логические цепочки: {len(result['logical_chains'])}")
    
    # Выводим финальный статус
    print(f"\n--- Финальный статус системы ---")
    status = system.get_system_status()
    for key, value in status.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_enhanced_agi()