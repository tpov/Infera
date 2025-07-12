import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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
    connections: Set[str] = field(default_factory=set)
    semantic_vector: Optional[List[float]] = None
    emotional_tone: float = 0.0
    logical_complexity: float = 0.5
    hidden_meaning: Optional[str] = None
    command_chain: List[str] = field(default_factory=list)

@dataclass
class EnhancedSystemState:
    """Расширенное состояние системы с пониманием контекста"""
    # Психическое состояние
    mental_state: MentalState = MentalState.NEUTRAL
    emotional_state: float = 0.0
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
    
    # Команды и их результаты
    command_history: List[Dict[str, Any]] = field(default_factory=list)
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Временные метки
    last_update: datetime = field(default_factory=datetime.now)
    session_start: datetime = field(default_factory=datetime.now)

# ============================================================================
# CONTEXTUAL COMMAND GENERATOR
# ============================================================================

class ContextualCommandGenerator:
    """Генератор команд с учетом контекста и скрытого смысла"""
    
    def __init__(self):
        self.command_templates = {
            'question': [
                "analyze_context context_type question",
                "create_hypothesis based_on {content}",
                "find_patterns in_context {content}",
                "extract_meaning from {content}",
                "build_logical_chain starting_with {content}"
            ],
            'statement': [
                "update_knowledge with {content}",
                "create_association between {content} and previous_context",
                "analyze_emotional_tone of {content}",
                "find_hidden_meaning in {content}",
                "synthesize_info from {content}"
            ],
            'command': [
                "execute_command {content}",
                "create_object based_on {content}",
                "modify_system according_to {content}",
                "configure_system with {content}",
                "activate_feature {content}"
            ],
            'observation': [
                "record_observation {content}",
                "analyze_pattern in {content}",
                "update_model with {content}",
                "create_hypothesis from {content}",
                "predict_outcome based_on {content}"
            ]
        }
        
        self.hidden_meaning_patterns = [
            (r'на самом деле (.+)', 'hidden_reality'),
            (r'по сути (.+)', 'essence'),
            (r'в действительности (.+)', 'actual_reality'),
            (r'скрыто (.+)', 'concealed'),
            (r'между строк (.+)', 'between_lines'),
            (r'подтекст (.+)', 'subtext')
        ]
    
    def generate_commands(self, context_analysis: Dict[str, Any], 
                         system_state: EnhancedSystemState) -> List[str]:
        """Генерирует команды на основе анализа контекста"""
        
        commands = []
        content = context_analysis.get('content', '')
        context_type = context_analysis.get('context_type', 'statement')
        
        # Базовые команды по типу контекста
        if context_type in self.command_templates:
            template = random.choice(self.command_templates[context_type])
            command = template.format(content=content)
            commands.append(command)
        
        # Команды для скрытого смысла
        hidden_meaning = context_analysis.get('hidden_meaning')
        if hidden_meaning:
            commands.append(f"extract_hidden_meaning {hidden_meaning}")
            commands.append(f"analyze_subtext {hidden_meaning}")
        
        # Команды для эмоционального анализа
        emotional_tone = context_analysis.get('emotional_tone', 0.0)
        if abs(emotional_tone) > 0.3:
            commands.append(f"analyze_emotion tone {emotional_tone}")
            commands.append(f"adjust_response_style emotional {emotional_tone}")
        
        # Команды для логической сложности
        logical_complexity = context_analysis.get('logical_complexity', 0.5)
        if logical_complexity > 0.7:
            commands.append(f"build_complex_logical_chain complexity {logical_complexity}")
            commands.append(f"analyze_logical_structure complexity {logical_complexity}")
        
        # Команды для вопросов
        if context_analysis.get('is_question', False):
            commands.append("generate_question_response")
            commands.append("find_relevant_knowledge")
            commands.append("create_explanation")
        
        return commands

# ============================================================================
# CONTEXT ANALYZER
# ============================================================================

class ContextAnalyzer:
    """Анализатор контекста для понимания скрытого смысла"""
    
    def __init__(self):
        self.emotional_keywords = {
            'positive': ['хорошо', 'отлично', 'прекрасно', 'радость', 'счастье', 'любовь', 'восторг'],
            'negative': ['плохо', 'ужасно', 'грусть', 'злость', 'разочарование', 'страх', 'отчаяние'],
            'neutral': ['обычно', 'нормально', 'стандартно', 'обыкновенно']
        }
        
        self.question_indicators = ['что', 'как', 'почему', 'когда', 'где', 'кто', '?', 'зачем', 'откуда']
        self.command_indicators = ['сделай', 'выполни', 'создай', 'удали', 'измени', 'настрой', 'включи']
        self.observation_indicators = ['вижу', 'наблюдаю', 'замечаю', 'видно', 'очевидно', 'явно']
        
        self.hidden_meaning_patterns = [
            (r'на самом деле (.+)', 'hidden_reality'),
            (r'по сути (.+)', 'essence'),
            (r'в действительности (.+)', 'actual_reality'),
            (r'скрыто (.+)', 'concealed'),
            (r'между строк (.+)', 'between_lines'),
            (r'подтекст (.+)', 'subtext'),
            (r'имеется в виду (.+)', 'implied'),
            (r'подразумевается (.+)', 'implied')
        ]
        
    def analyze(self, text: str, system_state: EnhancedSystemState) -> Dict[str, Any]:
        """Анализирует контекст текста"""
        
        text_lower = text.lower()
        
        # Определяем тип контекста
        context_type = self._determine_context_type(text_lower)
        
        # Анализируем эмоциональный тон
        emotional_tone = self._analyze_emotional_tone(text_lower)
        
        # Определяем логическую сложность
        logical_complexity = self._analyze_logical_complexity(text)
        
        # Ищем скрытый смысл
        hidden_meaning = self._find_hidden_meaning(text)
        
        # Определяем, является ли это вопросом
        is_question = any(indicator in text_lower for indicator in self.question_indicators)
        
        # Анализируем семантические связи
        semantic_connections = self._find_semantic_connections(text, system_state)
        
        return {
            'content': text,
            'context_type': context_type,
            'emotional_tone': emotional_tone,
            'logical_complexity': logical_complexity,
            'hidden_meaning': hidden_meaning,
            'is_question': is_question,
            'semantic_connections': semantic_connections,
            'text_length': len(text),
            'word_count': len(text.split()),
            'confidence': self._calculate_confidence(text, system_state)
        }
    
    def _determine_context_type(self, text: str) -> str:
        """Определяет тип контекста"""
        if any(indicator in text for indicator in self.question_indicators):
            return 'question'
        elif any(indicator in text for indicator in self.command_indicators):
            return 'command'
        elif any(indicator in text for indicator in self.observation_indicators):
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
        words = text.split()
        complexity = min(1.0, len(words) / 20.0)
        
        # Учитываем наличие логических связок
        logical_connectors = ['потому что', 'следовательно', 'поэтому', 'если', 'то', 'иначе', 'однако', 'но']
        connector_count = sum(1 for connector in logical_connectors if connector in text)
        complexity += connector_count * 0.1
        
        # Учитываем наличие подчинительных союзов
        subordinating_conjunctions = ['когда', 'где', 'куда', 'откуда', 'почему', 'зачем', 'как']
        sub_conj_count = sum(1 for conj in subordinating_conjunctions if conj in text)
        complexity += sub_conj_count * 0.05
        
        return min(1.0, complexity)
    
    def _find_hidden_meaning(self, text: str) -> Optional[str]:
        """Ищет скрытый смысл в тексте"""
        for pattern, meaning_type in self.hidden_meaning_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{meaning_type}: {match.group(1)}"
        
        return None
    
    def _find_semantic_connections(self, text: str, system_state: EnhancedSystemState) -> List[str]:
        """Находит семантические связи с предыдущим контекстом"""
        connections = []
        
        # Ищем связи с предыдущими контекстами
        for context_id, context_node in system_state.context_graph.items():
            if context_node.content.lower() in text.lower() or text.lower() in context_node.content.lower():
                connections.append(f"semantic_link:{context_id}")
        
        return connections
    
    def _calculate_confidence(self, text: str, system_state: EnhancedSystemState) -> float:
        """Вычисляет уверенность в анализе"""
        confidence = 0.5  # Базовая уверенность
        
        # Увеличиваем уверенность при наличии четких индикаторов
        if any(indicator in text.lower() for indicator in self.question_indicators):
            confidence += 0.2
        if any(indicator in text.lower() for indicator in self.command_indicators):
            confidence += 0.2
        if any(indicator in text.lower() for indicator in self.observation_indicators):
            confidence += 0.2
        
        # Учитываем длину текста
        if len(text.split()) > 10:
            confidence += 0.1
        
        return min(1.0, confidence)

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
            (r'(.+) следовательно (.+)', 'inference'),
            (r'(.+) поэтому (.+)', 'conclusion'),
            (r'(.+) в результате (.+)', 'result'),
            (r'(.+) из-за (.+)', 'cause')
        ]
        
        self.logical_operators = ['и', 'или', 'но', 'однако', 'хотя', 'несмотря на']
    
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
                    elif logic_type == 'explanation':
                        chain.append(f"Факт: {match.group(1)}")
                        chain.append(f"Объяснение: {match.group(2)}")
        
        return chain
    
    def find_patterns(self, data: List[str]) -> List[str]:
        """Ищет паттерны в данных"""
        patterns = []
        
        # Ищем повторяющиеся элементы
        word_freq = defaultdict(int)
        for item in data:
            words = item.lower().split()
            for word in words:
                if len(word) > 3:  # Игнорируем короткие слова
                    word_freq[word] += 1
        
        # Находим частые слова
        for word, freq in word_freq.items():
            if freq > 2:
                patterns.append(f"Частое слово: {word} (встречается {freq} раз)")
        
        # Ищем логические паттерны
        for item in data:
            for pattern, logic_type in self.causal_patterns:
                if re.search(pattern, item, re.IGNORECASE):
                    patterns.append(f"Логический паттерн: {logic_type}")
        
        return patterns

# ============================================================================
# INTEGRATED AGI CONTROLLER
# ============================================================================

class IntegratedAGIController:
    """Интегрированный AGI контроллер с пониманием контекста"""
    
    def __init__(self):
        self.state = EnhancedSystemState()
        self.command_generator = ContextualCommandGenerator()
        self.context_analyzer = ContextAnalyzer()
        self.logic_engine = LogicEngine()
        
        # Интеграция с существующим контроллером
        from command_controller import CommandController
        self.command_controller = CommandController()
        
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Обрабатывает вход пользователя с полным пониманием контекста"""
        
        start_time = time.time()
        
        # 1. Анализируем контекст
        context_analysis = self.context_analyzer.analyze(user_input, self.state)
        
        # 2. Обновляем психическое состояние
        self._update_mental_state(context_analysis)
        
        # 3. Генерируем команды с учетом контекста
        contextual_commands = self.command_generator.generate_commands(context_analysis, self.state)
        
        # 4. Преобразуем в команды для существующего контроллера
        system_commands = self._convert_to_system_commands(contextual_commands, context_analysis)
        
        # 5. Выполняем команды
        execution_result = self.command_controller.execute_commands(system_commands)
        
        # 6. Обновляем логические цепочки
        self._update_logical_chains(user_input, context_analysis, execution_result)
        
        # 7. Генерируем интеллектуальный ответ
        response = self._generate_intelligent_response(context_analysis, execution_result)
        
        # 8. Обновляем контекстную память
        self._update_context_memory(user_input, context_analysis, execution_result)
        
        processing_time = time.time() - start_time
        
        return {
            "response": response,
            "mental_state": self.state.mental_state.value,
            "confidence": self.state.confidence_level,
            "context_analysis": context_analysis,
            "contextual_commands": contextual_commands,
            "system_commands": system_commands,
            "execution_result": {
                "success": execution_result.success,
                "message": execution_result.message,
                "created_objects": len(execution_result.created_objects),
                "modified_objects": len(execution_result.modified_objects)
            },
            "logical_chains": self.state.logical_chains[-3:],
            "hidden_patterns": self.state.hidden_patterns[-2:],
            "processing_time": processing_time,
            "system_state_summary": self.get_system_state_summary()
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
        
        # Обновляем креативность
        logical_complexity = context_analysis.get('logical_complexity', 0.5)
        if logical_complexity > 0.7:
            self.state.creativity_level = min(1.0, self.state.creativity_level + 0.1)
        
        # Определяем психическое состояние
        if self.state.curiosity_level > 0.7:
            self.state.mental_state = MentalState.CURIOUS
        elif self.state.confidence_level < 0.3:
            self.state.mental_state = MentalState.CONFUSED
        elif self.state.confidence_level > 0.8:
            self.state.mental_state = MentalState.CONFIDENT
        elif logical_complexity > 0.7:
            self.state.mental_state = MentalState.LOGICAL
        elif abs(emotional_tone) > 0.5:
            self.state.mental_state = MentalState.EMOTIONAL
        else:
            self.state.mental_state = MentalState.NEUTRAL
    
    def _convert_to_system_commands(self, contextual_commands: List[str], 
                                  context_analysis: Dict[str, Any]) -> str:
        """Преобразует контекстные команды в команды системы"""
        
        system_commands = []
        
        for command in contextual_commands:
            if "analyze_context" in command:
                system_commands.append("create system quantity 1 type analyzer")
            elif "create_hypothesis" in command:
                system_commands.append("create hypothesis quantity 1 confidence 0.7")
            elif "find_patterns" in command:
                system_commands.append("create pattern_detector quantity 1 sensitivity high")
            elif "extract_meaning" in command:
                system_commands.append("create meaning_extractor quantity 1 depth deep")
            elif "build_logical_chain" in command:
                system_commands.append("create logic_chain quantity 1 complexity high")
            elif "update_knowledge" in command:
                system_commands.append("create knowledge_base quantity 1 update_mode active")
            elif "create_association" in command:
                system_commands.append("create association_network quantity 1 connections multiple")
            elif "analyze_emotion" in command:
                system_commands.append("create emotion_analyzer quantity 1 sensitivity high")
            elif "execute_command" in command:
                system_commands.append("create executor quantity 1 mode active")
            elif "record_observation" in command:
                system_commands.append("create observer quantity 1 observation_type detailed")
            else:
                # Общая команда для неизвестных команд
                system_commands.append("create system quantity 1 type general")
        
        return f"[{', '.join(system_commands)}]"
    
    def _update_logical_chains(self, user_input: str, context_analysis: Dict[str, Any], 
                              execution_result: Any):
        """Обновляет логические цепочки"""
        
        # Создаем новый контекстный узел
        chain_id = f"chain_{len(self.state.logical_chains)}"
        
        context_node = ContextNode(
            id=chain_id,
            content=user_input,
            context_type=ContextType.STATEMENT,
            confidence=context_analysis.get('confidence', 0.5),
            importance=0.7,
            emotional_tone=context_analysis.get('emotional_tone', 0.0),
            logical_complexity=context_analysis.get('logical_complexity', 0.5),
            hidden_meaning=context_analysis.get('hidden_meaning')
        )
        
        self.state.context_graph[chain_id] = context_node
        self.state.active_contexts.add(chain_id)
        self.state.context_history.append(chain_id)
        
        # Обновляем логические цепочки
        if len(self.state.logical_chains) > 0:
            self.state.logical_chains[-1].append(chain_id)
        else:
            self.state.logical_chains.append([chain_id])
        
        # Ищем паттерны
        if len(self.state.context_history) > 5:
            recent_contexts = list(self.state.context_history)[-5:]
            patterns = self.logic_engine.find_patterns([self.state.context_graph[ctx].content for ctx in recent_contexts])
            self.state.hidden_patterns.extend(patterns)
    
    def _generate_intelligent_response(self, context_analysis: Dict[str, Any], 
                                     execution_result: Any) -> str:
        """Генерирует интеллектуальный ответ"""
        
        response_parts = []
        
        # Базовый ответ в зависимости от типа контекста
        if context_analysis.get('is_question', False):
            if self.state.mental_state == MentalState.CURIOUS:
                response_parts.append("Это очень интересный вопрос!")
            elif self.state.mental_state == MentalState.CONFUSED:
                response_parts.append("Хм, это сложный вопрос. Давайте разберем его по частям.")
            else:
                response_parts.append("Позвольте мне проанализировать ваш вопрос.")
        else:
            if self.state.mental_state == MentalState.CONFIDENT:
                response_parts.append("Понимаю! Это важная информация.")
            else:
                response_parts.append("Спасибо за информацию. Я это учту.")
        
        # Добавляем информацию о скрытом смысле
        hidden_meaning = context_analysis.get('hidden_meaning')
        if hidden_meaning:
            response_parts.append(f"Я заметил скрытый смысл: {hidden_meaning}")
        
        # Добавляем информацию о логической сложности
        logical_complexity = context_analysis.get('logical_complexity', 0.5)
        if logical_complexity > 0.7:
            response_parts.append("Это довольно сложная логическая структура.")
        
        # Добавляем информацию о выполнении команд
        if execution_result.success:
            response_parts.append("Команды выполнены успешно.")
        else:
            response_parts.append("Возникли некоторые проблемы при выполнении команд.")
        
        return " ".join(response_parts)
    
    def _update_context_memory(self, user_input: str, context_analysis: Dict[str, Any], 
                              execution_result: Any):
        """Обновляет контекстную память"""
        
        # Сохраняем команды в истории
        self.state.command_history.append({
            'input': user_input,
            'context_analysis': context_analysis,
            'execution_result': {
                'success': execution_result.success,
                'message': execution_result.message
            },
            'timestamp': datetime.now()
        })
        
        # Обновляем семантическую сеть
        words = user_input.lower().split()
        for word in words:
            if word not in self.state.semantic_network:
                self.state.semantic_network[word] = set()
            
            # Связываем с другими словами в том же контексте
            for other_word in words:
                if other_word != word:
                    self.state.semantic_network[word].add(other_word)
    
    def get_system_state_summary(self) -> Dict[str, Any]:
        """Возвращает краткое описание состояния системы"""
        return {
            "mental_state": self.state.mental_state.value,
            "emotional_state": self.state.emotional_state,
            "confidence_level": self.state.confidence_level,
            "curiosity_level": self.state.curiosity_level,
            "creativity_level": self.state.creativity_level,
            "active_contexts": len(self.state.active_contexts),
            "logical_chains": len(self.state.logical_chains),
            "hidden_patterns": len(self.state.hidden_patterns),
            "context_history_length": len(self.state.context_history),
            "command_history_length": len(self.state.command_history),
            "semantic_network_size": len(self.state.semantic_network)
        }

# ============================================================================
# TESTING
# ============================================================================

def test_integrated_agi():
    """Тестирует интегрированную AGI систему"""
    print("Тестирование интегрированной AGI системы...")
    
    controller = IntegratedAGIController()
    
    # Тестовые запросы с разными типами контекста
    test_inputs = [
        "Что такое искусственный интеллект и как он работает?",
        "Мне грустно сегодня, на самом деле я чувствую себя одиноко",
        "Если идет дождь, то улицы мокрые, следовательно нужно взять зонт",
        "Создай систему автоматизации для управления домом",
        "Почему небо голубое? Это очень интересный вопрос",
        "Я вижу, что система работает нестабильно",
        "По сути, все сводится к пониманию контекста"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{'='*60}")
        print(f"ТЕСТ {i}: {user_input}")
        print(f"{'='*60}")
        
        result = controller.process_input(user_input)
        
        print(f"Ответ: {result['response']}")
        print(f"Психическое состояние: {result['mental_state']}")
        print(f"Уверенность: {result['confidence']:.2f}")
        print(f"Контекстные команды: {result['contextual_commands']}")
        print(f"Системные команды: {result['system_commands']}")
        print(f"Логические цепочки: {len(result['logical_chains'])}")
        print(f"Скрытые паттерны: {len(result['hidden_patterns'])}")
        print(f"Время обработки: {result['processing_time']:.3f} сек")
        
        # Выводим анализ контекста
        context = result['context_analysis']
        print(f"\nАНАЛИЗ КОНТЕКСТА:")
        print(f"  Тип: {context['context_type']}")
        print(f"  Эмоциональный тон: {context['emotional_tone']:.2f}")
        print(f"  Логическая сложность: {context['logical_complexity']:.2f}")
        print(f"  Это вопрос: {context['is_question']}")
        if context.get('hidden_meaning'):
            print(f"  Скрытый смысл: {context['hidden_meaning']}")
    
    # Выводим финальный статус
    print(f"\n{'='*60}")
    print("ФИНАЛЬНЫЙ СТАТУС СИСТЕМЫ")
    print(f"{'='*60}")
    status = controller.get_system_state_summary()
    for key, value in status.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_integrated_agi()