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
# УНИВЕРСАЛЬНАЯ СИСТЕМА КОМАНД
# ============================================================================

class CommandType(Enum):
    """Универсальные типы команд"""
    CREATE = "create"      # Создать что-то
    MODIFY = "modify"      # Изменить что-то
    DELETE = "delete"      # Удалить что-то
    GET = "get"           # Получить что-то
    SET = "set"           # Установить что-то
    FIND = "find"         # Найти что-то
    ANALYZE = "analyze"   # Проанализировать что-то
    COMPARE = "compare"   # Сравнить что-то
    CONNECT = "connect"   # Связать что-то
    SEPARATE = "separate" # Разделить что-то
    TRANSFORM = "transform" # Преобразовать что-то
    PREDICT = "predict"   # Предсказать что-то
    EXPLAIN = "explain"   # Объяснить что-то
    LEARN = "learn"       # Научиться чему-то
    REMEMBER = "remember" # Запомнить что-то
    FORGET = "forget"     # Забыть что-то
    THINK = "think"       # Подумать о чем-то
    FEEL = "feel"         # Чувствовать что-то
    UNDERSTAND = "understand" # Понять что-то

@dataclass
class UniversalCommand:
    """Универсальная команда, которая может описать любое действие"""
    command_type: CommandType
    subject: str           # Что (объект, концепция, идея)
    action: str           # Что делать
    target: Optional[str] = None  # На что воздействовать
    parameters: Dict[str, Any] = field(default_factory=dict)  # Дополнительные параметры
    context: Dict[str, Any] = field(default_factory=dict)    # Контекст выполнения
    confidence: float = 0.5  # Уверенность в команде
    priority: float = 0.5    # Приоритет выполнения
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class UniversalState:
    """Универсальное состояние системы"""
    # Динамические объекты (создаются в процессе работы)
    objects: Dict[str, Any] = field(default_factory=dict)
    
    # Концепции и идеи
    concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Связи между объектами и концепциями
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    
    # Эмоциональное и когнитивное состояние
    emotional_state: Dict[str, float] = field(default_factory=dict)
    cognitive_state: Dict[str, float] = field(default_factory=dict)
    
    # Память и опыт
    memory: List[Dict[str, Any]] = field(default_factory=list)
    experience: Dict[str, Any] = field(default_factory=dict)
    
    # Текущий контекст
    current_context: Dict[str, Any] = field(default_factory=dict)
    
    # История команд
    command_history: List[UniversalCommand] = field(default_factory=list)
    
    # Временные метки
    last_update: datetime = field(default_factory=datetime.now)

# ============================================================================
# УНИВЕРСАЛЬНЫЙ АНАЛИЗАТОР КОНТЕКСТА
# ============================================================================

class UniversalContextAnalyzer:
    """Анализатор, который понимает любой контекст без предопределенных словарей"""
    
    def __init__(self):
        # Универсальные паттерны для извлечения смысла
        self.patterns = {
            'action': [
                r'(\w+)\s+(созда|дела|стро|пиш|чита|дума|поним|зна|виж|слыш|чувств|хот|нуж|мож|долж)',
                r'(созда|дела|стро|пиш|чита|дума|поним|зна|виж|слыш|чувств|хот|нуж|мож|долж)\s+(\w+)',
                r'(\w+)\s+(есть|был|будет|стал|станет)',
                r'(есть|был|будет|стал|станет)\s+(\w+)'
            ],
            'relationship': [
                r'(\w+)\s+(и|или|но|однако|хотя|потому что|поэтому|следовательно)\s+(\w+)',
                r'(\w+)\s+(больше|меньше|равно|похож|отличается от)\s+(\w+)',
                r'(\w+)\s+(внутри|снаружи|рядом с|далеко от|близко к)\s+(\w+)'
            ],
            'emotion': [
                r'(рад|груст|зл|страш|удиви|интерес|скуч|весел|сердит|спокоен)',
                r'(люблю|ненавижу|боюсь|надеюсь|сомневаюсь|уверен|неуверен)'
            ],
            'question': [
                r'(что|как|почему|когда|где|кто|зачем|откуда|куда|сколько|какой|какая|какое|какие)\s+(\w+)',
                r'(\w+)\s+\?'
            ],
            'comparison': [
                r'(\w+)\s+(как|похож на|отличается от|лучше|хуже|больше|меньше)\s+(\w+)'
            ]
        }
    
    def analyze(self, text: str, state: UniversalState) -> Dict[str, Any]:
        """Анализирует любой текст и извлекает универсальные команды"""
        
        analysis = {
            'text': text,
            'commands': [],
            'concepts': [],
            'relationships': [],
            'emotions': [],
            'questions': [],
            'comparisons': [],
            'confidence': 0.5,
            'complexity': 0.5
        }
        
        # Извлекаем все возможные паттерны
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if pattern_type == 'action':
                        analysis['commands'].append({
                            'type': 'action',
                            'subject': match.group(1) if match.group(1) else 'unknown',
                            'action': match.group(2) if match.group(2) else 'unknown',
                            'confidence': 0.7
                        })
                    elif pattern_type == 'relationship':
                        analysis['relationships'].append({
                            'subject': match.group(1),
                            'relation': match.group(2),
                            'object': match.group(3) if len(match.groups()) > 2 else 'unknown'
                        })
                    elif pattern_type == 'emotion':
                        analysis['emotions'].append({
                            'emotion': match.group(1),
                            'intensity': 0.8
                        })
                    elif pattern_type == 'question':
                        analysis['questions'].append({
                            'question_word': match.group(1),
                            'subject': match.group(2) if len(match.groups()) > 1 else 'unknown'
                        })
                    elif pattern_type == 'comparison':
                        analysis['comparisons'].append({
                            'subject': match.group(1),
                            'comparison': match.group(2),
                            'object': match.group(3) if len(match.groups()) > 2 else 'unknown'
                        })
        
        # Извлекаем концепции (слова, которые могут быть важными)
        words = text.lower().split()
        for word in words:
            if len(word) > 3 and word not in ['этот', 'тот', 'какой', 'такой', 'все', 'всех']:
                analysis['concepts'].append({
                    'concept': word,
                    'frequency': words.count(word),
                    'importance': len(word) / 10.0
                })
        
        # Вычисляем сложность
        analysis['complexity'] = min(1.0, len(words) / 20.0 + len(analysis['commands']) * 0.1)
        
        # Вычисляем уверенность
        analysis['confidence'] = min(1.0, 0.5 + len(analysis['commands']) * 0.1 + len(analysis['concepts']) * 0.05)
        
        return analysis

# ============================================================================
# УНИВЕРСАЛЬНЫЙ ГЕНЕРАТОР КОМАНД
# ============================================================================

class UniversalCommandGenerator:
    """Генератор универсальных команд на основе анализа контекста"""
    
    def __init__(self):
        self.command_mappings = {
            'созда': CommandType.CREATE,
            'дела': CommandType.CREATE,
            'стро': CommandType.CREATE,
            'пиш': CommandType.CREATE,
            'измен': CommandType.MODIFY,
            'меня': CommandType.MODIFY,
            'удал': CommandType.DELETE,
            'убира': CommandType.DELETE,
            'получ': CommandType.GET,
            'найди': CommandType.FIND,
            'ищи': CommandType.FIND,
            'анализ': CommandType.ANALYZE,
            'изуч': CommandType.ANALYZE,
            'сравн': CommandType.COMPARE,
            'свяж': CommandType.CONNECT,
            'раздел': CommandType.SEPARATE,
            'преобраз': CommandType.TRANSFORM,
            'предскаж': CommandType.PREDICT,
            'объясн': CommandType.EXPLAIN,
            'науч': CommandType.LEARN,
            'запомн': CommandType.REMEMBER,
            'забуд': CommandType.FORGET,
            'дума': CommandType.THINK,
            'чувств': CommandType.FEEL,
            'поним': CommandType.UNDERSTAND
        }
    
    def generate_commands(self, analysis: Dict[str, Any], state: UniversalState) -> List[UniversalCommand]:
        """Генерирует универсальные команды на основе анализа"""
        
        commands = []
        
        # Генерируем команды из действий
        for action in analysis.get('commands', []):
            command_type = self._map_action_to_command_type(action['action'])
            if command_type:
                command = UniversalCommand(
                    command_type=command_type,
                    subject=action['subject'],
                    action=action['action'],
                    confidence=action['confidence'],
                    context={'source': 'action_analysis'}
                )
                commands.append(command)
        
        # Генерируем команды из вопросов
        for question in analysis.get('questions', []):
            command = UniversalCommand(
                command_type=CommandType.FIND,
                subject=question['subject'],
                action='найти_ответ',
                target=question['question_word'],
                confidence=0.8,
                context={'source': 'question_analysis'}
            )
            commands.append(command)
        
        # Генерируем команды из эмоций
        for emotion in analysis.get('emotions', []):
            command = UniversalCommand(
                command_type=CommandType.FEEL,
                subject='система',
                action='испытать_эмоцию',
                target=emotion['emotion'],
                parameters={'intensity': emotion['intensity']},
                confidence=0.6,
                context={'source': 'emotion_analysis'}
            )
            commands.append(command)
        
        # Генерируем команды из сравнений
        for comparison in analysis.get('comparisons', []):
            command = UniversalCommand(
                command_type=CommandType.COMPARE,
                subject=comparison['subject'],
                action='сравнить',
                target=comparison['object'],
                parameters={'comparison_type': comparison['comparison']},
                confidence=0.7,
                context={'source': 'comparison_analysis'}
            )
            commands.append(command)
        
        # Генерируем команды из концепций
        for concept in analysis.get('concepts', []):
            if concept['importance'] > 0.5:
                command = UniversalCommand(
                    command_type=CommandType.UNDERSTAND,
                    subject=concept['concept'],
                    action='изучить_концепцию',
                    parameters={'importance': concept['importance']},
                    confidence=concept['importance'],
                    context={'source': 'concept_analysis'}
                )
                commands.append(command)
        
        return commands
    
    def _map_action_to_command_type(self, action: str) -> Optional[CommandType]:
        """Сопоставляет действие с типом команды"""
        action_lower = action.lower()
        for action_pattern, command_type in self.command_mappings.items():
            if action_pattern in action_lower:
                return command_type
        return None

# ============================================================================
# УНИВЕРСАЛЬНЫЙ КОНТРОЛЛЕР
# ============================================================================

class UniversalController:
    """Универсальный контроллер, который может обработать любую команду"""
    
    def __init__(self):
        self.state = UniversalState()
        self.analyzer = UniversalContextAnalyzer()
        self.command_generator = UniversalCommandGenerator()
        
        # Интеграция с существующим контроллером
        from command_controller import CommandController
        self.command_controller = CommandController()
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Обрабатывает любой пользовательский ввод"""
        
        start_time = time.time()
        
        # 1. Анализируем контекст
        analysis = self.analyzer.analyze(user_input, self.state)
        
        # 2. Генерируем универсальные команды
        universal_commands = self.command_generator.generate_commands(analysis, self.state)
        
        # 3. Преобразуем в системные команды
        system_commands = self._convert_to_system_commands(universal_commands, analysis)
        
        # 4. Выполняем команды
        execution_result = self.command_controller.execute_commands(system_commands)
        
        # 5. Обновляем состояние
        self._update_state(user_input, analysis, universal_commands, execution_result)
        
        # 6. Генерируем ответ
        response = self._generate_response(analysis, universal_commands, execution_result)
        
        processing_time = time.time() - start_time
        
        return {
            "response": response,
            "analysis": analysis,
            "universal_commands": [cmd.__dict__ for cmd in universal_commands],
            "system_commands": system_commands,
            "execution_result": {
                "success": execution_result.success,
                "message": execution_result.message,
                "created_objects": len(execution_result.created_objects),
                "modified_objects": len(execution_result.modified_objects)
            },
            "processing_time": processing_time,
            "state_summary": self.get_state_summary()
        }
    
    def _convert_to_system_commands(self, universal_commands: List[UniversalCommand], 
                                   analysis: Dict[str, Any]) -> str:
        """Преобразует универсальные команды в системные команды"""
        
        system_commands = []
        
        for cmd in universal_commands:
            if cmd.command_type == CommandType.CREATE:
                system_commands.append(f"create {cmd.subject} quantity 1 type {cmd.action}")
            elif cmd.command_type == CommandType.MODIFY:
                system_commands.append(f"set {cmd.subject} {cmd.action} value active")
            elif cmd.command_type == CommandType.DELETE:
                system_commands.append(f"delete {cmd.subject}")
            elif cmd.command_type == CommandType.GET:
                system_commands.append(f"get {cmd.subject} status value active")
            elif cmd.command_type == CommandType.FIND:
                system_commands.append(f"create finder quantity 1 target {cmd.subject}")
            elif cmd.command_type == CommandType.ANALYZE:
                system_commands.append(f"create analyzer quantity 1 target {cmd.subject}")
            elif cmd.command_type == CommandType.COMPARE:
                system_commands.append(f"create comparator quantity 1 subject {cmd.subject} target {cmd.target}")
            elif cmd.command_type == CommandType.CONNECT:
                system_commands.append(f"create connector quantity 1 subject {cmd.subject} target {cmd.target}")
            elif cmd.command_type == CommandType.TRANSFORM:
                system_commands.append(f"create transformer quantity 1 input {cmd.subject} output {cmd.target}")
            elif cmd.command_type == CommandType.PREDICT:
                system_commands.append(f"create predictor quantity 1 based_on {cmd.subject}")
            elif cmd.command_type == CommandType.EXPLAIN:
                system_commands.append(f"create explainer quantity 1 subject {cmd.subject}")
            elif cmd.command_type == CommandType.LEARN:
                system_commands.append(f"create learner quantity 1 subject {cmd.subject}")
            elif cmd.command_type == CommandType.REMEMBER:
                system_commands.append(f"create memory quantity 1 content {cmd.subject}")
            elif cmd.command_type == CommandType.THINK:
                system_commands.append(f"create thinker quantity 1 topic {cmd.subject}")
            elif cmd.command_type == CommandType.FEEL:
                system_commands.append(f"create emotion_processor quantity 1 emotion {cmd.target}")
            elif cmd.command_type == CommandType.UNDERSTAND:
                system_commands.append(f"create understanding_engine quantity 1 concept {cmd.subject}")
            else:
                # Универсальная команда для неизвестных типов
                system_commands.append(f"create universal_processor quantity 1 action {cmd.action} subject {cmd.subject}")
        
        return f"[{', '.join(system_commands)}]"
    
    def _update_state(self, user_input: str, analysis: Dict[str, Any], 
                     commands: List[UniversalCommand], execution_result: Any):
        """Обновляет состояние системы"""
        
        # Добавляем команды в историю
        self.state.command_history.extend(commands)
        
        # Обновляем память
        self.state.memory.append({
            'input': user_input,
            'analysis': analysis,
            'commands': [cmd.__dict__ for cmd in commands],
            'execution_result': {
                'success': execution_result.success,
                'message': execution_result.message
            },
            'timestamp': datetime.now()
        })
        
        # Обновляем концепции
        for concept in analysis.get('concepts', []):
            concept_name = concept['concept']
            if concept_name not in self.state.concepts:
                self.state.concepts[concept_name] = {
                    'frequency': 0,
                    'importance': concept['importance'],
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now()
                }
            else:
                self.state.concepts[concept_name]['frequency'] += 1
                self.state.concepts[concept_name]['last_seen'] = datetime.now()
        
        # Обновляем эмоциональное состояние
        for emotion in analysis.get('emotions', []):
            emotion_name = emotion['emotion']
            self.state.emotional_state[emotion_name] = emotion['intensity']
        
        # Обновляем когнитивное состояние
        self.state.cognitive_state['complexity'] = analysis.get('complexity', 0.5)
        self.state.cognitive_state['confidence'] = analysis.get('confidence', 0.5)
        
        # Обновляем текущий контекст
        self.state.current_context = {
            'last_input': user_input,
            'last_analysis': analysis,
            'last_commands_count': len(commands),
            'timestamp': datetime.now()
        }
        
        self.state.last_update = datetime.now()
    
    def _generate_response(self, analysis: Dict[str, Any], commands: List[UniversalCommand], 
                          execution_result: Any) -> str:
        """Генерирует ответ на основе анализа и команд"""
        
        response_parts = []
        
        # Анализируем тип взаимодействия
        if analysis.get('questions'):
            response_parts.append("Я понимаю ваш вопрос. Давайте разберем это.")
        elif analysis.get('emotions'):
            response_parts.append("Я чувствую эмоциональный контекст вашего сообщения.")
        elif analysis.get('commands'):
            response_parts.append("Я понял, что нужно сделать.")
        else:
            response_parts.append("Я обработал вашу информацию.")
        
        # Добавляем информацию о найденных концепциях
        concepts = analysis.get('concepts', [])
        if concepts:
            important_concepts = [c['concept'] for c in concepts if c['importance'] > 0.7]
            if important_concepts:
                response_parts.append(f"Я выделил важные концепции: {', '.join(important_concepts[:3])}")
        
        # Добавляем информацию о выполнении команд
        if execution_result.success:
            response_parts.append("Команды выполнены успешно.")
        else:
            response_parts.append("Возникли некоторые сложности при выполнении.")
        
        # Добавляем информацию о сложности
        complexity = analysis.get('complexity', 0.5)
        if complexity > 0.7:
            response_parts.append("Это довольно сложная задача.")
        
        return " ".join(response_parts)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Возвращает краткое описание состояния"""
        return {
            "total_objects": len(self.state.objects),
            "total_concepts": len(self.state.concepts),
            "command_history_length": len(self.state.command_history),
            "memory_length": len(self.state.memory),
            "emotional_states": len(self.state.emotional_state),
            "cognitive_states": len(self.state.cognitive_state),
            "last_update": self.state.last_update.isoformat()
        }

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

def test_universal_system():
    """Тестирует универсальную систему команд"""
    print("Тестирование универсальной системы команд...")
    
    controller = UniversalController()
    
    # Тестовые запросы - любые, какие придут в голову
    test_inputs = [
        "Мне грустно",
        "Что такое любовь?",
        "Создай робота",
        "Сравни кошку и собаку",
        "Я хочу пиццу",
        "Почему небо голубое?",
        "Научи меня программировать",
        "Запомни, что я люблю музыку",
        "Думаю о будущем",
        "Понимаю, что жизнь сложная",
        "Чувствую радость",
        "Анализирую ситуацию",
        "Предскажи погоду",
        "Объясни квантовую физику",
        "Свяжи эти идеи вместе"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{'='*60}")
        print(f"ТЕСТ {i}: {user_input}")
        print(f"{'='*60}")
        
        result = controller.process_input(user_input)
        
        print(f"Ответ: {result['response']}")
        print(f"Универсальные команды: {len(result['universal_commands'])}")
        for cmd in result['universal_commands'][:3]:  # Показываем первые 3
            print(f"  - {cmd['command_type']}: {cmd['action']} {cmd['subject']}")
        
        print(f"Системные команды: {result['system_commands']}")
        print(f"Время обработки: {result['processing_time']:.3f} сек")
        
        # Выводим анализ
        analysis = result['analysis']
        print(f"\nАНАЛИЗ:")
        print(f"  Команды: {len(analysis['commands'])}")
        print(f"  Концепции: {len(analysis['concepts'])}")
        print(f"  Эмоции: {len(analysis['emotions'])}")
        print(f"  Вопросы: {len(analysis['questions'])}")
        print(f"  Сложность: {analysis['complexity']:.2f}")
        print(f"  Уверенность: {analysis['confidence']:.2f}")
    
    # Выводим финальный статус
    print(f"\n{'='*60}")
    print("ФИНАЛЬНЫЙ СТАТУС СИСТЕМЫ")
    print(f"{'='*60}")
    status = controller.get_state_summary()
    for key, value in status.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_universal_system()