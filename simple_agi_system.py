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
# ПРОСТАЯ AGI СИСТЕМА
# ============================================================================

class CommandType(Enum):
    """Типы команд"""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    GET = "get"
    FIND = "find"
    ANALYZE = "analyze"
    COMPARE = "compare"
    CONNECT = "connect"
    TRANSFORM = "transform"
    PREDICT = "predict"
    EXPLAIN = "explain"
    LEARN = "learn"
    REMEMBER = "remember"
    THINK = "think"
    FEEL = "feel"
    UNDERSTAND = "understand"

@dataclass
class SimpleCommand:
    """Простая команда"""
    command_type: CommandType
    subject: str
    action: str
    target: Optional[str] = None
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SimpleState:
    """Простое состояние системы"""
    objects: Dict[str, Any] = field(default_factory=dict)
    concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    emotions: Dict[str, float] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    command_history: List[SimpleCommand] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)

class SimpleAnalyzer:
    """Простой анализатор контекста"""
    
    def __init__(self):
        self.patterns = {
            'action': [
                r'(\w+)\s+(созда|дела|стро|пиш|чита|дума|поним|зна|виж|слыш|чувств|хот|нуж|мож|долж)',
                r'(созда|дела|стро|пиш|чита|дума|поним|зна|виж|слыш|чувств|хот|нуж|мож|долж)\s+(\w+)'
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
    
    def analyze(self, text: str, state: SimpleState) -> Dict[str, Any]:
        """Анализирует текст"""
        
        analysis = {
            'text': text,
            'commands': [],
            'concepts': [],
            'emotions': [],
            'questions': [],
            'comparisons': [],
            'confidence': 0.5,
            'complexity': 0.5
        }
        
        # Извлекаем паттерны
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
        
        # Извлекаем концепции
        words = text.lower().split()
        for word in words:
            if len(word) > 3 and word not in ['этот', 'тот', 'какой', 'такой', 'все', 'всех']:
                analysis['concepts'].append({
                    'concept': word,
                    'frequency': words.count(word),
                    'importance': len(word) / 10.0
                })
        
        # Вычисляем сложность и уверенность
        analysis['complexity'] = min(1.0, len(words) / 20.0 + len(analysis['commands']) * 0.1)
        analysis['confidence'] = min(1.0, 0.5 + len(analysis['commands']) * 0.1 + len(analysis['concepts']) * 0.05)
        
        return analysis

class SimpleCommandGenerator:
    """Простой генератор команд"""
    
    def __init__(self):
        self.command_mappings = {
            'созда': CommandType.CREATE,
            'дела': CommandType.CREATE,
            'стро': CommandType.CREATE,
            'измен': CommandType.MODIFY,
            'удал': CommandType.DELETE,
            'найди': CommandType.FIND,
            'анализ': CommandType.ANALYZE,
            'сравн': CommandType.COMPARE,
            'свяж': CommandType.CONNECT,
            'преобраз': CommandType.TRANSFORM,
            'предскаж': CommandType.PREDICT,
            'объясн': CommandType.EXPLAIN,
            'науч': CommandType.LEARN,
            'запомн': CommandType.REMEMBER,
            'дума': CommandType.THINK,
            'чувств': CommandType.FEEL,
            'поним': CommandType.UNDERSTAND
        }
    
    def generate_commands(self, analysis: Dict[str, Any], state: SimpleState) -> List[SimpleCommand]:
        """Генерирует команды"""
        
        commands = []
        
        # Команды из действий
        for action in analysis.get('commands', []):
            command_type = self._map_action_to_command_type(action['action'])
            if command_type:
                command = SimpleCommand(
                    command_type=command_type,
                    subject=action['subject'],
                    action=action['action'],
                    confidence=action['confidence']
                )
                commands.append(command)
        
        # Команды из вопросов
        for question in analysis.get('questions', []):
            command = SimpleCommand(
                command_type=CommandType.FIND,
                subject=question['subject'],
                action='найти_ответ',
                target=question['question_word'],
                confidence=0.8
            )
            commands.append(command)
        
        # Команды из эмоций
        for emotion in analysis.get('emotions', []):
            command = SimpleCommand(
                command_type=CommandType.FEEL,
                subject='система',
                action='испытать_эмоцию',
                target=emotion['emotion'],
                confidence=0.6
            )
            commands.append(command)
        
        # Команды из концепций
        for concept in analysis.get('concepts', []):
            if concept['importance'] > 0.5:
                command = SimpleCommand(
                    command_type=CommandType.UNDERSTAND,
                    subject=concept['concept'],
                    action='изучить_концепцию',
                    confidence=concept['importance']
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

class SimpleController:
    """Простой контроллер"""
    
    def __init__(self):
        self.state = SimpleState()
        self.analyzer = SimpleAnalyzer()
        self.command_generator = SimpleCommandGenerator()
        
        # Интеграция с существующим контроллером
        from command_controller import CommandController
        self.command_controller = CommandController()
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Обрабатывает ввод"""
        
        start_time = time.time()
        
        # 1. Анализируем контекст
        analysis = self.analyzer.analyze(user_input, self.state)
        
        # 2. Генерируем команды
        commands = self.command_generator.generate_commands(analysis, self.state)
        
        # 3. Преобразуем в системные команды
        system_commands = self._convert_to_system_commands(commands, analysis)
        
        # 4. Выполняем команды
        execution_result = self.command_controller.execute_commands(system_commands)
        
        # 5. Обновляем состояние
        self._update_state(user_input, analysis, commands, execution_result)
        
        # 6. Генерируем ответ
        response = self._generate_response(analysis, commands, execution_result)
        
        processing_time = time.time() - start_time
        
        return {
            "response": response,
            "analysis": analysis,
            "commands": [cmd.__dict__ for cmd in commands],
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
    
    def _convert_to_system_commands(self, commands: List[SimpleCommand], analysis: Dict[str, Any]) -> str:
        """Преобразует команды в системные"""
        
        system_commands = []
        
        for cmd in commands:
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
                system_commands.append(f"create universal_processor quantity 1 action {cmd.action} subject {cmd.subject}")
        
        return f"[{', '.join(system_commands)}]"
    
    def _update_state(self, user_input: str, analysis: Dict[str, Any], commands: List[SimpleCommand], execution_result: Any):
        """Обновляет состояние"""
        
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
        
        # Обновляем эмоции
        for emotion in analysis.get('emotions', []):
            emotion_name = emotion['emotion']
            self.state.emotions[emotion_name] = emotion['intensity']
        
        self.state.last_update = datetime.now()
    
    def _generate_response(self, analysis: Dict[str, Any], commands: List[SimpleCommand], execution_result: Any) -> str:
        """Генерирует ответ"""
        
        response_parts = []
        
        # Базовый ответ
        if analysis.get('questions'):
            response_parts.append("Я понимаю ваш вопрос. Давайте разберем это.")
        elif analysis.get('emotions'):
            response_parts.append("Я чувствую эмоциональный контекст вашего сообщения.")
        elif analysis.get('commands'):
            response_parts.append("Я понял, что нужно сделать.")
        else:
            response_parts.append("Я обработал вашу информацию.")
        
        # Информация о концепциях
        concepts = analysis.get('concepts', [])
        if concepts:
            important_concepts = [c['concept'] for c in concepts if c['importance'] > 0.7]
            if important_concepts:
                response_parts.append(f"Я выделил важные концепции: {', '.join(important_concepts[:3])}")
        
        # Информация о выполнении
        if execution_result.success:
            response_parts.append("Команды выполнены успешно.")
        else:
            response_parts.append("Возникли некоторые сложности при выполнении.")
        
        return " ".join(response_parts)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Возвращает сводку состояния"""
        return {
            "total_objects": len(self.state.objects),
            "total_concepts": len(self.state.concepts),
            "command_history_length": len(self.state.command_history),
            "memory_length": len(self.state.memory),
            "emotional_states": len(self.state.emotions),
            "last_update": self.state.last_update.isoformat()
        }

# ============================================================================
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# ============================================================================

def interactive_mode():
    """Интерактивный режим"""
    
    print("Добро пожаловать в простую AGI систему!")
    print("Введите любой текст, и система обработает его.")
    print("Для выхода введите 'выход' или 'exit'")
    print("Для получения статистики введите 'статистика'")
    print("-" * 80)
    
    controller = SimpleController()
    
    while True:
        try:
            user_input = input("\nВы: ").strip()
            
            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("До свидания!")
                break
            
            elif user_input.lower() == 'статистика':
                summary = controller.get_state_summary()
                print(f"\nСТАТИСТИКА СИСТЕМЫ:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
                continue
            
            elif not user_input:
                continue
            
            # Обрабатываем ввод
            result = controller.process_input(user_input)
            
            print(f"\nAGI: {result['response']}")
            print(f"Команд сгенерировано: {len(result['commands'])}")
            print(f"Время обработки: {result['processing_time']:.3f} сек")
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

def test_simple_agi():
    """Тестирует простую AGI систему"""
    print("Тестирование простой AGI системы...")
    
    controller = SimpleController()
    
    # Тестовые запросы
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
        print(f"Команд: {len(result['commands'])}")
        for j, cmd in enumerate(result['commands'][:3]):
            print(f"  {j+1}. {cmd['command_type']}: {cmd['action']} {cmd['subject']}")
        print(f"Время: {result['processing_time']:.3f} сек")
        
        # Выводим анализ
        analysis = result['analysis']
        print(f"\nАНАЛИЗ:")
        print(f"  Команды: {len(analysis['commands'])}")
        print(f"  Концепции: {len(analysis['concepts'])}")
        print(f"  Эмоции: {len(analysis['emotions'])}")
        print(f"  Вопросы: {len(analysis['questions'])}")
        print(f"  Сложность: {analysis['complexity']:.2f}")
        print(f"  Уверенность: {analysis['confidence']:.2f}")
    
    # Финальная статистика
    print(f"\n{'='*60}")
    print("ФИНАЛЬНАЯ СТАТИСТИКА")
    print(f"{'='*60}")
    
    summary = controller.get_state_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        test_simple_agi()