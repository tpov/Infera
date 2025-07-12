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
# СИСТЕМА ЦИКЛИЧЕСКОГО ОБУЧЕНИЯ
# ============================================================================

@dataclass
class LearningCycle:
    """Один цикл обучения"""
    cycle_id: str
    input_text: str
    initial_analysis: Dict[str, Any]
    generated_commands: List[Dict[str, Any]]
    execution_result: Dict[str, Any]
    feedback: Dict[str, Any]
    improved_commands: List[Dict[str, Any]]
    final_result: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    learning_score: float = 0.0

@dataclass
class CyclicLearningState:
    """Состояние системы циклического обучения"""
    # История циклов обучения
    learning_cycles: List[LearningCycle] = field(default_factory=list)
    
    # Паттерны обучения
    learned_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Улучшения команд
    command_improvements: Dict[str, List[str]] = field(default_factory=dict)
    
    # Статистика обучения
    total_cycles: int = 0
    successful_cycles: int = 0
    average_learning_score: float = 0.0
    
    # Текущий контекст обучения
    current_learning_context: Dict[str, Any] = field(default_factory=dict)
    
    # Временные метки
    last_learning_update: datetime = field(default_factory=datetime.now)

class CyclicLearningSystem:
    """Система циклического обучения для улучшения понимания контекста"""
    
    def __init__(self, max_cycles: int = 10):
        self.max_cycles = max_cycles
        self.state = CyclicLearningState()
        
        # Импортируем универсальный контроллер
        from universal_command_system import UniversalController
        self.controller = UniversalController()
        
        # Система обратной связи
        self.feedback_system = FeedbackSystem()
        
        # Система улучшения команд
        self.command_improver = CommandImprover()
    
    def learn_from_input(self, user_input: str) -> Dict[str, Any]:
        """Обучается на входе пользователя через циклы"""
        
        print(f"\n{'='*80}")
        print(f"НАЧАЛО ОБУЧЕНИЯ: {user_input}")
        print(f"{'='*80}")
        
        cycles = []
        current_input = user_input
        
        for cycle_num in range(self.max_cycles):
            print(f"\n--- ЦИКЛ ОБУЧЕНИЯ {cycle_num + 1} ---")
            
            # 1. Анализируем текущий ввод
            analysis = self.controller.analyzer.analyze(current_input, self.controller.state)
            
            # 2. Генерируем команды
            commands = self.controller.command_generator.generate_commands(analysis, self.controller.state)
            
            # 3. Выполняем команды
            execution_result = self._execute_commands(commands)
            
            # 4. Получаем обратную связь
            feedback = self.feedback_system.analyze_feedback(
                user_input, analysis, commands, execution_result
            )
            
            # 5. Улучшаем команды на основе обратной связи
            improved_commands = self.command_improver.improve_commands(
                commands, feedback, self.state
            )
            
            # 6. Выполняем улучшенные команды
            final_result = self._execute_commands(improved_commands)
            
            # 7. Создаем цикл обучения
            cycle = LearningCycle(
                cycle_id=f"cycle_{cycle_num + 1}",
                input_text=current_input,
                initial_analysis=analysis,
                generated_commands=[cmd.__dict__ for cmd in commands],
                execution_result=execution_result,
                feedback=feedback,
                improved_commands=[cmd.__dict__ for cmd in improved_commands],
                final_result=final_result,
                learning_score=self._calculate_learning_score(feedback, final_result)
            )
            
            cycles.append(cycle)
            
            # 8. Обновляем состояние обучения
            self._update_learning_state(cycle)
            
            # 9. Генерируем новый ввод для следующего цикла
            current_input = self._generate_next_cycle_input(cycle)
            
            print(f"Цикл {cycle_num + 1} завершен. Оценка обучения: {cycle.learning_score:.3f}")
            
            # Проверяем, нужно ли остановить обучение
            if self._should_stop_learning(cycle):
                print(f"Останавливаем обучение на цикле {cycle_num + 1}")
                break
        
        # Сохраняем все циклы
        self.state.learning_cycles.extend(cycles)
        self.state.total_cycles += len(cycles)
        
        # Вычисляем общую статистику
        self._update_learning_statistics()
        
        return {
            "original_input": user_input,
            "total_cycles": len(cycles),
            "cycles": [cycle.__dict__ for cycle in cycles],
            "final_result": cycles[-1].final_result if cycles else None,
            "learning_statistics": self._get_learning_statistics(),
            "learned_patterns": self.state.learned_patterns
        }
    
    def _execute_commands(self, commands: List[Any]) -> Dict[str, Any]:
        """Выполняет команды и возвращает результат"""
        try:
            # Преобразуем в системные команды
            system_commands = self.controller._convert_to_system_commands(commands, {})
            
            # Выполняем через контроллер
            execution_result = self.controller.command_controller.execute_commands(system_commands)
            
            return {
                "success": execution_result.success,
                "message": execution_result.message,
                "created_objects": len(execution_result.created_objects),
                "modified_objects": len(execution_result.modified_objects),
                "warnings": execution_result.warnings,
                "contradictions": execution_result.contradictions
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Ошибка выполнения: {str(e)}",
                "created_objects": 0,
                "modified_objects": 0,
                "warnings": [str(e)],
                "contradictions": []
            }
    
    def _calculate_learning_score(self, feedback: Dict[str, Any], final_result: Dict[str, Any]) -> float:
        """Вычисляет оценку обучения"""
        score = 0.0
        
        # Базовый балл за успешное выполнение
        if final_result.get('success', False):
            score += 0.3
        
        # Баллы за обратную связь
        feedback_score = feedback.get('overall_score', 0.0)
        score += feedback_score * 0.4
        
        # Баллы за создание объектов
        created_objects = final_result.get('created_objects', 0)
        score += min(0.2, created_objects * 0.05)
        
        # Штраф за предупреждения и противоречия
        warnings = len(final_result.get('warnings', []))
        contradictions = len(final_result.get('contradictions', []))
        score -= (warnings + contradictions) * 0.05
        
        return max(0.0, min(1.0, score))
    
    def _should_stop_learning(self, cycle: LearningCycle) -> bool:
        """Определяет, нужно ли остановить обучение"""
        
        # Останавливаем, если достигли высокого качества
        if cycle.learning_score > 0.8:
            return True
        
        # Останавливаем, если нет улучшений в последних циклах
        if len(self.state.learning_cycles) >= 3:
            recent_scores = [c.learning_score for c in self.state.learning_cycles[-3:]]
            if max(recent_scores) - min(recent_scores) < 0.1:
                return True
        
        return False
    
    def _generate_next_cycle_input(self, cycle: LearningCycle) -> str:
        """Генерирует ввод для следующего цикла обучения"""
        
        # Анализируем обратную связь
        feedback = cycle.feedback
        
        if feedback.get('needs_clarification', False):
            return f"Уточни: {cycle.input_text}"
        elif feedback.get('needs_more_context', False):
            return f"Добавь контекст к: {cycle.input_text}"
        elif feedback.get('needs_simplification', False):
            return f"Упрости: {cycle.input_text}"
        else:
            # Генерируем улучшенную версию
            return self._improve_input(cycle.input_text, feedback)
    
    def _improve_input(self, original_input: str, feedback: Dict[str, Any]) -> str:
        """Улучшает ввод на основе обратной связи"""
        
        improvements = []
        
        if feedback.get('missing_emotion', False):
            improvements.append("с эмоциональным контекстом")
        
        if feedback.get('missing_logic', False):
            improvements.append("с логической структурой")
        
        if feedback.get('missing_action', False):
            improvements.append("с конкретным действием")
        
        if improvements:
            return f"{original_input} ({', '.join(improvements)})"
        else:
            return original_input
    
    def _update_learning_state(self, cycle: LearningCycle):
        """Обновляет состояние обучения"""
        
        # Обновляем паттерны обучения
        pattern_key = f"pattern_{len(self.state.learned_patterns)}"
        self.state.learned_patterns[pattern_key] = {
            'input_pattern': cycle.input_text,
            'successful_commands': cycle.improved_commands,
            'learning_score': cycle.learning_score,
            'feedback_insights': cycle.feedback
        }
        
        # Обновляем улучшения команд
        for cmd in cycle.improved_commands:
            cmd_type = cmd.get('command_type', 'unknown')
            if cmd_type not in self.state.command_improvements:
                self.state.command_improvements[cmd_type] = []
            
            improvement = f"Улучшение для {cmd_type}: {cmd.get('action', 'unknown')}"
            self.state.command_improvements[cmd_type].append(improvement)
        
        # Обновляем контекст обучения
        self.state.current_learning_context = {
            'last_cycle_score': cycle.learning_score,
            'last_feedback': cycle.feedback,
            'total_cycles': len(self.state.learning_cycles) + 1
        }
        
        self.state.last_learning_update = datetime.now()
    
    def _update_learning_statistics(self):
        """Обновляет статистику обучения"""
        if self.state.learning_cycles:
            scores = [cycle.learning_score for cycle in self.state.learning_cycles]
            self.state.average_learning_score = sum(scores) / len(scores)
            self.state.successful_cycles = len([s for s in scores if s > 0.7])
    
    def _get_learning_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику обучения"""
        return {
            "total_cycles": self.state.total_cycles,
            "successful_cycles": self.state.successful_cycles,
            "average_learning_score": self.state.average_learning_score,
            "learned_patterns_count": len(self.state.learned_patterns),
            "command_improvements_count": len(self.state.command_improvements)
        }

# ============================================================================
# СИСТЕМА ОБРАТНОЙ СВЯЗИ
# ============================================================================

class FeedbackSystem:
    """Система анализа обратной связи для улучшения команд"""
    
    def __init__(self):
        self.feedback_patterns = {
            'success_indicators': ['успешно', 'правильно', 'хорошо', 'отлично', 'верно'],
            'failure_indicators': ['ошибка', 'неправильно', 'плохо', 'неверно', 'неудача'],
            'clarification_needed': ['неясно', 'непонятно', 'уточни', 'объясни', 'что имеется в виду'],
            'context_needed': ['мало информации', 'нужно больше', 'не хватает контекста'],
            'simplification_needed': ['слишком сложно', 'упрости', 'проще']
        }
    
    def analyze_feedback(self, original_input: str, analysis: Dict[str, Any], 
                        commands: List[Any], execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Анализирует обратную связь и предлагает улучшения"""
        
        feedback = {
            'overall_score': 0.0,
            'success_indicators': [],
            'failure_indicators': [],
            'needs_clarification': False,
            'needs_more_context': False,
            'needs_simplification': False,
            'missing_emotion': False,
            'missing_logic': False,
            'missing_action': False,
            'suggestions': []
        }
        
        # Анализируем результат выполнения
        if execution_result.get('success', False):
            feedback['overall_score'] += 0.4
            feedback['success_indicators'].append('Команды выполнены успешно')
        else:
            feedback['failure_indicators'].append('Ошибка выполнения команд')
        
        # Анализируем созданные объекты
        created_objects = execution_result.get('created_objects', 0)
        if created_objects > 0:
            feedback['overall_score'] += 0.2
            feedback['success_indicators'].append(f'Создано объектов: {created_objects}')
        else:
            feedback['failure_indicators'].append('Не создано объектов')
        
        # Анализируем предупреждения и противоречия
        warnings = execution_result.get('warnings', [])
        contradictions = execution_result.get('contradictions', [])
        
        if warnings:
            feedback['overall_score'] -= 0.1
            feedback['failure_indicators'].extend(warnings)
        
        if contradictions:
            feedback['overall_score'] -= 0.2
            feedback['failure_indicators'].extend(contradictions)
        
        # Анализируем команды
        self._analyze_commands(commands, feedback)
        
        # Анализируем исходный ввод
        self._analyze_input(original_input, analysis, feedback)
        
        # Генерируем предложения
        self._generate_suggestions(feedback)
        
        return feedback
    
    def _analyze_commands(self, commands: List[Any], feedback: Dict[str, Any]):
        """Анализирует сгенерированные команды"""
        
        if not commands:
            feedback['missing_action'] = True
            feedback['suggestions'].append('Добавить конкретные действия')
            return
        
        # Анализируем типы команд
        command_types = [cmd.command_type for cmd in commands]
        
        if CommandType.FEEL not in command_types and any('эмоц' in str(cmd) for cmd in commands):
            feedback['missing_emotion'] = True
            feedback['suggestions'].append('Добавить обработку эмоций')
        
        if CommandType.ANALYZE not in command_types and CommandType.UNDERSTAND not in command_types:
            feedback['missing_logic'] = True
            feedback['suggestions'].append('Добавить логический анализ')
        
        # Проверяем разнообразие команд
        if len(set(command_types)) < 2:
            feedback['suggestions'].append('Добавить больше разнообразия в команды')
    
    def _analyze_input(self, original_input: str, analysis: Dict[str, Any], feedback: Dict[str, Any]):
        """Анализирует исходный ввод"""
        
        # Проверяем наличие эмоций
        if not analysis.get('emotions'):
            feedback['missing_emotion'] = True
            feedback['suggestions'].append('Добавить эмоциональный контекст')
        
        # Проверяем логическую сложность
        complexity = analysis.get('complexity', 0.5)
        if complexity < 0.3:
            feedback['needs_more_context'] = True
            feedback['suggestions'].append('Добавить больше контекста')
        elif complexity > 0.8:
            feedback['needs_simplification'] = True
            feedback['suggestions'].append('Упростить выражение')
        
        # Проверяем наличие действий
        if not analysis.get('commands'):
            feedback['missing_action'] = True
            feedback['suggestions'].append('Добавить конкретные действия')
    
    def _generate_suggestions(self, feedback: Dict[str, Any]):
        """Генерирует предложения по улучшению"""
        
        if feedback['missing_emotion']:
            feedback['suggestions'].append('Использовать команды типа FEEL для обработки эмоций')
        
        if feedback['missing_logic']:
            feedback['suggestions'].append('Использовать команды типа ANALYZE или UNDERSTAND')
        
        if feedback['missing_action']:
            feedback['suggestions'].append('Использовать команды типа CREATE, MODIFY или FIND')
        
        if feedback['needs_clarification']:
            feedback['suggestions'].append('Добавить команды типа EXPLAIN для уточнения')
        
        if feedback['needs_more_context']:
            feedback['suggestions'].append('Использовать команды типа REMEMBER для сохранения контекста')

# ============================================================================
# СИСТЕМА УЛУЧШЕНИЯ КОМАНД
# ============================================================================

class CommandImprover:
    """Система улучшения команд на основе обратной связи"""
    
    def __init__(self):
        self.improvement_strategies = {
            'missing_emotion': self._add_emotion_commands,
            'missing_logic': self._add_logic_commands,
            'missing_action': self._add_action_commands,
            'needs_clarification': self._add_clarification_commands,
            'needs_more_context': self._add_context_commands
        }
    
    def improve_commands(self, original_commands: List[Any], feedback: Dict[str, Any], 
                        learning_state: CyclicLearningState) -> List[Any]:
        """Улучшает команды на основе обратной связи"""
        
        improved_commands = original_commands.copy()
        
        # Применяем стратегии улучшения
        for issue, strategy in self.improvement_strategies.items():
            if feedback.get(issue, False):
                new_commands = strategy(original_commands, feedback, learning_state)
                improved_commands.extend(new_commands)
        
        # Улучшаем существующие команды
        for i, cmd in enumerate(improved_commands):
            improved_commands[i] = self._improve_single_command(cmd, feedback)
        
        return improved_commands
    
    def _add_emotion_commands(self, commands: List[Any], feedback: Dict[str, Any], 
                             learning_state: CyclicLearningState) -> List[Any]:
        """Добавляет команды для обработки эмоций"""
        from universal_command_system import UniversalCommand, CommandType
        
        emotion_commands = []
        
        # Добавляем команду для анализа эмоций
        emotion_cmd = UniversalCommand(
            command_type=CommandType.FEEL,
            subject='система',
            action='анализировать_эмоции',
            target='входной_текст',
            confidence=0.8,
            context={'source': 'emotion_improvement'}
        )
        emotion_commands.append(emotion_cmd)
        
        return emotion_commands
    
    def _add_logic_commands(self, commands: List[Any], feedback: Dict[str, Any], 
                           learning_state: CyclicLearningState) -> List[Any]:
        """Добавляет команды для логического анализа"""
        from universal_command_system import UniversalCommand, CommandType
        
        logic_commands = []
        
        # Добавляем команду для логического анализа
        logic_cmd = UniversalCommand(
            command_type=CommandType.ANALYZE,
            subject='логическая_структура',
            action='анализировать_логику',
            confidence=0.8,
            context={'source': 'logic_improvement'}
        )
        logic_commands.append(logic_cmd)
        
        return logic_commands
    
    def _add_action_commands(self, commands: List[Any], feedback: Dict[str, Any], 
                            learning_state: CyclicLearningState) -> List[Any]:
        """Добавляет команды для конкретных действий"""
        from universal_command_system import UniversalCommand, CommandType
        
        action_commands = []
        
        # Добавляем универсальную команду действия
        action_cmd = UniversalCommand(
            command_type=CommandType.CREATE,
            subject='действие',
            action='выполнить_действие',
            confidence=0.7,
            context={'source': 'action_improvement'}
        )
        action_commands.append(action_cmd)
        
        return action_commands
    
    def _add_clarification_commands(self, commands: List[Any], feedback: Dict[str, Any], 
                                   learning_state: CyclicLearningState) -> List[Any]:
        """Добавляет команды для уточнения"""
        from universal_command_system import UniversalCommand, CommandType
        
        clarification_commands = []
        
        # Добавляем команду для объяснения
        explain_cmd = UniversalCommand(
            command_type=CommandType.EXPLAIN,
            subject='неясные_понятия',
            action='объяснить_понятия',
            confidence=0.8,
            context={'source': 'clarification_improvement'}
        )
        clarification_commands.append(explain_cmd)
        
        return clarification_commands
    
    def _add_context_commands(self, commands: List[Any], feedback: Dict[str, Any], 
                             learning_state: CyclicLearningState) -> List[Any]:
        """Добавляет команды для работы с контекстом"""
        from universal_command_system import UniversalCommand, CommandType
        
        context_commands = []
        
        # Добавляем команду для запоминания контекста
        remember_cmd = UniversalCommand(
            command_type=CommandType.REMEMBER,
            subject='контекст',
            action='запомнить_контекст',
            confidence=0.8,
            context={'source': 'context_improvement'}
        )
        context_commands.append(remember_cmd)
        
        return context_commands
    
    def _improve_single_command(self, command: Any, feedback: Dict[str, Any]) -> Any:
        """Улучшает отдельную команду"""
        
        # Увеличиваем уверенность, если команда успешна
        if feedback.get('overall_score', 0) > 0.7:
            command.confidence = min(1.0, command.confidence + 0.1)
        
        # Добавляем контекст из обратной связи
        if feedback.get('suggestions'):
            command.context['feedback_suggestions'] = feedback['suggestions']
        
        return command

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

def test_cyclic_learning():
    """Тестирует систему циклического обучения"""
    print("Тестирование системы циклического обучения...")
    
    learning_system = CyclicLearningSystem(max_cycles=5)
    
    # Тестовые запросы для обучения
    test_inputs = [
        "Мне грустно и я хочу поговорить",
        "Что такое искусственный интеллект и как он работает?",
        "Создай робота, который может думать",
        "Сравни кошку и собаку, объясни различия",
        "Научи меня программировать на Python"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{'='*80}")
        print(f"ОБУЧЕНИЕ {i}: {user_input}")
        print(f"{'='*80}")
        
        result = learning_system.learn_from_input(user_input)
        
        print(f"Всего циклов: {result['total_cycles']}")
        print(f"Финальный результат: {result['final_result']}")
        
        # Выводим статистику обучения
        stats = result['learning_statistics']
        print(f"\nСТАТИСТИКА ОБУЧЕНИЯ:")
        print(f"  Средняя оценка: {stats['average_learning_score']:.3f}")
        print(f"  Успешных циклов: {stats['successful_cycles']}")
        print(f"  Изученных паттернов: {stats['learned_patterns_count']}")
        
        # Выводим изученные паттерны
        if result['learned_patterns']:
            print(f"\nИЗУЧЕННЫЕ ПАТТЕРНЫ:")
            for pattern_id, pattern in result['learned_patterns'].items():
                print(f"  {pattern_id}: {pattern['input_pattern']} (оценка: {pattern['learning_score']:.3f})")
    
    # Выводим общую статистику
    print(f"\n{'='*80}")
    print("ОБЩАЯ СТАТИСТИКА ОБУЧЕНИЯ")
    print(f"{'='*80}")
    
    total_stats = learning_system._get_learning_statistics()
    for key, value in total_stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_cyclic_learning()