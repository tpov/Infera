from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import random
import re
from enum import Enum

# Импортируем созданные системы
from state_compression_system import AdaptiveStateController, CompressedState
from hierarchical_command_system import ContextAwareController, HierarchicalCommand, CommandLevel
from cyclic_learning_system import CyclicLearningSystem, BasicQuery, QueryType

class AGIMode(Enum):
    STATE_COMPRESSION = "state_compression"
    HIERARCHICAL = "hierarchical"
    CYCLIC_LEARNING = "cyclic_learning"
    HYBRID = "hybrid"

@dataclass
class AGIProcessingResult:
    """Результат обработки AGI системы"""
    mode: AGIMode
    user_query: str
    processing_time: float
    confidence: float
    generated_commands: List[str]
    system_state: Dict[str, Any]
    neural_response: str
    context_path: List[str]
    learning_cycles: Optional[List[Any]] = None
    state_description: Optional[str] = None

class IntegratedAGISystem:
    """Интегрированная AGI система"""
    
    def __init__(self):
        # Инициализируем все подсистемы
        self.state_controller = AdaptiveStateController()
        self.hierarchical_controller = ContextAwareController()
        self.cyclic_learning = CyclicLearningSystem(max_cycles=8)
        
        # Настройки системы
        self.current_mode = AGIMode.HYBRID
        self.processing_history = []
        self.confidence_threshold = 0.7
        
    def process_query(self, user_query: str, mode: Optional[AGIMode] = None) -> AGIProcessingResult:
        """Обрабатывает запрос пользователя"""
        
        start_time = datetime.now()
        
        # Определяем режим обработки
        if mode is None:
            mode = self._determine_optimal_mode(user_query)
        
        print(f"Обрабатываем запрос в режиме: {mode.value}")
        print(f"Запрос: {user_query}")
        
        # Обрабатываем в выбранном режиме
        if mode == AGIMode.STATE_COMPRESSION:
            result = self._process_with_state_compression(user_query)
        elif mode == AGIMode.HIERARCHICAL:
            result = self._process_with_hierarchical(user_query)
        elif mode == AGIMode.CYCLIC_LEARNING:
            result = self._process_with_cyclic_learning(user_query)
        elif mode == AGIMode.HYBRID:
            result = self._process_with_hybrid(user_query)
        else:
            raise ValueError(f"Неизвестный режим: {mode}")
        
        # Вычисляем время обработки
        processing_time = (datetime.now() - start_time).total_seconds()
        result.processing_time = processing_time
        
        # Сохраняем в историю
        self.processing_history.append(result)
        
        return result
    
    def _determine_optimal_mode(self, user_query: str) -> AGIMode:
        """Определяет оптимальный режим обработки"""
        
        query_lower = user_query.lower()
        
        # Анализируем сложность запроса
        complexity_score = self._calculate_complexity_score(user_query)
        
        # Анализируем тип запроса
        if any(word in query_lower for word in ['эмоции', 'чувства', 'настроение', 'состояние']):
            return AGIMode.STATE_COMPRESSION
        
        if any(word in query_lower for word in ['система', 'процесс', 'функция', 'структура']):
            return AGIMode.HIERARCHICAL
        
        if any(word in query_lower for word in ['было', 'стало', 'изменилось', 'эволюция']):
            return AGIMode.CYCLIC_LEARNING
        
        # По умолчанию используем гибридный режим
        return AGIMode.HYBRID
    
    def _calculate_complexity_score(self, query: str) -> float:
        """Вычисляет оценку сложности запроса"""
        score = 0.0
        
        # Количество слов
        word_count = len(query.split())
        score += min(word_count / 20.0, 1.0) * 0.3
        
        # Наличие специальных символов
        special_chars = len(re.findall(r'[^\w\s]', query))
        score += min(special_chars / 10.0, 1.0) * 0.2
        
        # Наличие чисел
        numbers = len(re.findall(r'\d+', query))
        score += min(numbers / 5.0, 1.0) * 0.2
        
        # Наличие логических операторов
        logical_ops = len(re.findall(r'\b(если|то|и|или|но|затем|потом)\b', query.lower()))
        score += min(logical_ops / 3.0, 1.0) * 0.3
        
        return min(score, 1.0)
    
    def _process_with_state_compression(self, user_query: str) -> AGIProcessingResult:
        """Обрабатывает запрос с компрессией состояний"""
        
        # Создаем входной вектор (симуляция)
        input_vector = self._create_input_vector(user_query)
        
        # Обновляем состояние
        state = self.state_controller.update_state(input_vector, user_query)
        
        # Генерируем команды из состояния
        commands = self.state_controller.generate_commands_from_state(state)
        
        # Выполняем команды (симуляция)
        system_state = self._execute_commands(commands)
        
        # Генерируем ответ
        neural_response = self._generate_state_based_response(state, system_state)
        
        return AGIProcessingResult(
            mode=AGIMode.STATE_COMPRESSION,
            user_query=user_query,
            processing_time=0.0,  # Будет установлено позже
            confidence=state.to_vector().mean().item(),
            generated_commands=[commands] if commands else [],
            system_state=system_state,
            neural_response=neural_response,
            context_path=[],
            state_description=state.to_text_description()
        )
    
    def _process_with_hierarchical(self, user_query: str) -> AGIProcessingResult:
        """Обрабатывает запрос с иерархическими командами"""
        
        # Обрабатываем через иерархический контроллер
        result = self.hierarchical_controller.process_query(user_query)
        
        # Извлекаем команды
        commands = [cmd.command for cmd in result["generated_commands"]]
        
        # Выполняем команды
        system_state = self._execute_commands(commands)
        
        return AGIProcessingResult(
            mode=AGIMode.HIERARCHICAL,
            user_query=user_query,
            processing_time=0.0,
            confidence=result["confidence_level"],
            generated_commands=commands,
            system_state=system_state,
            neural_response=result["final_response"] if "final_response" in result else "Обработано иерархически",
            context_path=result["context_path"]
        )
    
    def _process_with_cyclic_learning(self, user_query: str) -> AGIProcessingResult:
        """Обрабатывает запрос с циклическим обучением"""
        
        # Обрабатываем через циклическое обучение
        result = self.cyclic_learning.process_with_cyclic_learning(user_query)
        
        # Извлекаем команды из всех циклов
        all_commands = []
        for cycle in result["cycles"]:
            all_commands.extend(cycle.controller_commands)
        
        # Выполняем команды
        system_state = self._execute_commands(all_commands)
        
        return AGIProcessingResult(
            mode=AGIMode.CYCLIC_LEARNING,
            user_query=user_query,
            processing_time=0.0,
            confidence=result["overall_confidence"],
            generated_commands=all_commands,
            system_state=system_state,
            neural_response=result["final_response"],
            context_path=[],
            learning_cycles=result["cycles"]
        )
    
    def _process_with_hybrid(self, user_query: str) -> AGIProcessingResult:
        """Обрабатывает запрос в гибридном режиме"""
        
        # Запускаем все три режима параллельно
        state_result = self._process_with_state_compression(user_query)
        hierarchical_result = self._process_with_hierarchical(user_query)
        cyclic_result = self._process_with_cyclic_learning(user_query)
        
        # Объединяем результаты
        combined_commands = []
        combined_commands.extend(state_result.generated_commands)
        combined_commands.extend(hierarchical_result.generated_commands)
        combined_commands.extend(cyclic_result.generated_commands)
        
        # Вычисляем общую уверенность
        total_confidence = (
            state_result.confidence + 
            hierarchical_result.confidence + 
            cyclic_result.confidence
        ) / 3
        
        # Генерируем комбинированный ответ
        combined_response = self._combine_responses([
            state_result.neural_response,
            hierarchical_result.neural_response,
            cyclic_result.neural_response
        ])
        
        # Выполняем все команды
        system_state = self._execute_commands(combined_commands)
        
        return AGIProcessingResult(
            mode=AGIMode.HYBRID,
            user_query=user_query,
            processing_time=0.0,
            confidence=total_confidence,
            generated_commands=combined_commands,
            system_state=system_state,
            neural_response=combined_response,
            context_path=hierarchical_result.context_path
        )
    
    def _create_input_vector(self, query: str) -> Any:
        """Создает входной вектор (симуляция)"""
        # В реальности здесь должна быть векторизация
        import numpy as np
        return np.random.randn(768)
    
    def _execute_commands(self, commands: List[str]) -> Dict[str, Any]:
        """Выполняет команды (симуляция)"""
        system_state = {
            "total_objects": len(commands),
            "object_types": {},
            "execution_success": True,
            "warnings": [],
            "errors": []
        }
        
        for command in commands:
            if "create" in command:
                system_state["total_objects"] += 1
            elif "delete" in command:
                system_state["total_objects"] = max(0, system_state["total_objects"] - 1)
        
        return system_state
    
    def _generate_state_based_response(self, state: CompressedState, system_state: Dict[str, Any]) -> str:
        """Генерирует ответ на основе состояния"""
        response_parts = []
        
        # Добавляем описание состояния
        response_parts.append(f"Текущее состояние: {state.to_text_description()}")
        
        # Добавляем информацию о системе
        response_parts.append(f"Объектов в системе: {system_state['total_objects']}")
        
        return ". ".join(response_parts)
    
    def _combine_responses(self, responses: List[str]) -> str:
        """Объединяет несколько ответов в один"""
        if not responses:
            return "Нет ответа"
        
        # Убираем дубликаты и объединяем
        unique_responses = list(set(responses))
        return " | ".join(unique_responses)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Возвращает статус системы"""
        return {
            "current_mode": self.current_mode.value,
            "processing_history_length": len(self.processing_history),
            "confidence_threshold": self.confidence_threshold,
            "average_confidence": self._calculate_average_confidence(),
            "total_processed_queries": len(self.processing_history)
        }
    
    def _calculate_average_confidence(self) -> float:
        """Вычисляет среднюю уверенность"""
        if not self.processing_history:
            return 0.0
        
        total_confidence = sum(result.confidence for result in self.processing_history)
        return total_confidence / len(self.processing_history)
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Анализирует паттерны запросов"""
        if not self.processing_history:
            return {"message": "Нет данных для анализа"}
        
        # Анализируем режимы
        mode_counts = {}
        for result in self.processing_history:
            mode = result.mode.value
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # Анализируем уверенность
        confidence_by_mode = {}
        for result in self.processing_history:
            mode = result.mode.value
            if mode not in confidence_by_mode:
                confidence_by_mode[mode] = []
            confidence_by_mode[mode].append(result.confidence)
        
        # Вычисляем среднюю уверенность по режимам
        avg_confidence_by_mode = {}
        for mode, confidences in confidence_by_mode.items():
            avg_confidence_by_mode[mode] = sum(confidences) / len(confidences)
        
        return {
            "mode_distribution": mode_counts,
            "average_confidence_by_mode": avg_confidence_by_mode,
            "total_queries": len(self.processing_history)
        }

def test_integrated_agi_system():
    """Тестирует интегрированную AGI систему"""
    system = IntegratedAGISystem()
    
    test_queries = [
        "Я чувствую себя хорошо сегодня",
        "Создать систему автоматизации с 5 датчиками",
        "Было 3 яблока, стало 5 яблок",
        "Если температура высокая, то включить вентиляцию",
        "Система управления освещением в доме"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"ТЕСТИРОВАНИЕ AGI СИСТЕМЫ")
        print(f"Запрос: {query}")
        print(f"{'='*80}")
        
        # Обрабатываем запрос
        result = system.process_query(query)
        
        print(f"\nРЕЗУЛЬТАТЫ:")
        print(f"  Режим: {result.mode.value}")
        print(f"  Время обработки: {result.processing_time:.3f} сек")
        print(f"  Уверенность: {result.confidence:.3f}")
        print(f"  Сгенерировано команд: {len(result.generated_commands)}")
        print(f"  Ответ: {result.neural_response}")
        
        if result.context_path:
            print(f"  Путь контекста: {' -> '.join(result.context_path)}")
        
        if result.state_description:
            print(f"  Описание состояния: {result.state_description}")
    
    # Анализируем паттерны
    print(f"\n{'='*80}")
    print(f"АНАЛИЗ ПАТТЕРНОВ")
    print(f"{'='*80}")
    
    analysis = system.analyze_query_patterns()
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Статус системы
    print(f"\n{'='*80}")
    print(f"СТАТУС СИСТЕМЫ")
    print(f"{'='*80}")
    
    status = system.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_integrated_agi_system()