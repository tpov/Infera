from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from enum import Enum
import re

# Импортируем наши системы
from state_compression_system import AdaptiveStateController, CompressedState
from hierarchical_command_system import ContextAwareController, HierarchicalCommand
from cyclic_analysis_system import CyclicAnalysisEngine, CommandGeneratorFromAnalysis

class AGIArchitecture(Enum):
    STATE_COMPRESSION = "state_compression"
    HIERARCHICAL_COMMANDS = "hierarchical_commands"
    CYCLIC_ANALYSIS = "cyclic_analysis"
    HYBRID = "hybrid"

@dataclass
class AGIResponse:
    """Ответ AGI системы"""
    user_query: str
    architecture_used: AGIArchitecture
    commands_generated: List[str]
    state_description: str
    confidence: float
    analysis_insights: List[str]
    execution_results: Dict[str, Any]
    processing_time: float

class IntegratedAGISystem:
    """Интегрированная AGI система"""
    
    def __init__(self):
        # Инициализируем все подсистемы
        self.state_controller = AdaptiveStateController()
        self.hierarchical_controller = ContextAwareController()
        self.cyclic_analyzer = CommandGeneratorFromAnalysis()
        
        # История обработки
        self.processing_history = []
        self.current_architecture = AGIArchitecture.HYBRID
        
    def process_query(self, user_query: str, architecture: AGIArchitecture = AGIArchitecture.HYBRID) -> AGIResponse:
        """Обрабатывает запрос пользователя"""
        
        start_time = datetime.now()
        
        print(f"\n{'='*80}")
        print(f"AGI СИСТЕМА: Обработка запроса")
        print(f"Запрос: {user_query}")
        print(f"Архитектура: {architecture.value}")
        print(f"{'='*80}")
        
        # Выбираем архитектуру обработки
        if architecture == AGIArchitecture.STATE_COMPRESSION:
            result = self._process_with_state_compression(user_query)
        elif architecture == AGIArchitecture.HIERARCHICAL_COMMANDS:
            result = self._process_with_hierarchical_commands(user_query)
        elif architecture == AGIArchitecture.CYCLIC_ANALYSIS:
            result = self._process_with_cyclic_analysis(user_query)
        else:  # HYBRID
            result = self._process_with_hybrid_approach(user_query)
        
        # Вычисляем время обработки
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Создаем ответ
        response = AGIResponse(
            user_query=user_query,
            architecture_used=architecture,
            commands_generated=result['commands'],
            state_description=result.get('state_description', ''),
            confidence=result.get('confidence', 0.0),
            analysis_insights=result.get('insights', []),
            execution_results=result.get('execution_results', {}),
            processing_time=processing_time
        )
        
        # Сохраняем в историю
        self.processing_history.append(response)
        
        # Выводим результаты
        self._print_response(response)
        
        return response
    
    def _process_with_state_compression(self, user_query: str) -> Dict[str, Any]:
        """Обработка с компрессией состояний"""
        print("\n1. КОМПРЕССИЯ СОСТОЯНИЙ")
        print("-" * 40)
        
        # Создаем фиктивный вектор (в реальности это будет от векторизатора)
        import numpy as np
        input_vector = np.random.randn(768)
        
        # Обновляем состояние
        state = self.state_controller.update_state(input_vector, user_query)
        
        # Генерируем команды из состояния
        commands = self.state_controller.generate_commands_from_state(state)
        
        return {
            'commands': [commands] if commands else [],
            'state_description': self.state_controller.get_state_description(),
            'confidence': 0.8,
            'insights': ['Состояние сжато и адаптировано под запрос'],
            'execution_results': {'success': True, 'message': 'Состояние обновлено'}
        }
    
    def _process_with_hierarchical_commands(self, user_query: str) -> Dict[str, Any]:
        """Обработка с иерархическими командами"""
        print("\n2. ИЕРАРХИЧЕСКИЕ КОМАНДЫ")
        print("-" * 40)
        
        # Обрабатываем запрос через иерархический контроллер
        result = self.hierarchical_controller.process_query(user_query)
        
        # Извлекаем команды
        commands = [cmd.command for cmd in result['generated_commands']]
        
        return {
            'commands': commands,
            'state_description': f"Контекст: {' -> '.join(result['context_path'])}",
            'confidence': result['confidence_level'],
            'insights': [f"Создано {len(commands)} иерархических команд"],
            'execution_results': result['execution_results']
        }
    
    def _process_with_cyclic_analysis(self, user_query: str) -> Dict[str, Any]:
        """Обработка с циклическим анализом"""
        print("\n3. ЦИКЛИЧЕСКИЙ АНАЛИЗ")
        print("-" * 40)
        
        # Генерируем команды через циклический анализ
        commands = self.cyclic_analyzer.generate_commands_from_analysis(user_query)
        
        return {
            'commands': [commands] if commands else [],
            'state_description': 'Анализ завершен',
            'confidence': 0.7,
            'insights': ['Выполнен глубокий анализ с поиском скрытых смыслов'],
            'execution_results': {'success': True, 'message': 'Анализ выполнен'}
        }
    
    def _process_with_hybrid_approach(self, user_query: str) -> Dict[str, Any]:
        """Гибридная обработка"""
        print("\n4. ГИБРИДНАЯ ОБРАБОТКА")
        print("-" * 40)
        
        # Выполняем все три подхода
        state_result = self._process_with_state_compression(user_query)
        hierarchical_result = self._process_with_hierarchical_commands(user_query)
        cyclic_result = self._process_with_cyclic_analysis(user_query)
        
        # Объединяем результаты
        all_commands = []
        all_insights = []
        
        all_commands.extend(state_result['commands'])
        all_commands.extend(hierarchical_result['commands'])
        all_commands.extend(cyclic_result['commands'])
        
        all_insights.extend(state_result['insights'])
        all_insights.extend(hierarchical_result['insights'])
        all_insights.extend(cyclic_result['insights'])
        
        # Вычисляем среднюю уверенность
        avg_confidence = (
            state_result['confidence'] + 
            hierarchical_result['confidence'] + 
            cyclic_result['confidence']
        ) / 3
        
        return {
            'commands': all_commands,
            'state_description': f"Гибридное состояние: {state_result['state_description']} | {hierarchical_result['state_description']}",
            'confidence': avg_confidence,
            'insights': all_insights,
            'execution_results': {
                'state_compression': state_result['execution_results'],
                'hierarchical': hierarchical_result['execution_results'],
                'cyclic': cyclic_result['execution_results']
            }
        }
    
    def _print_response(self, response: AGIResponse):
        """Выводит результаты обработки"""
        print(f"\n{'='*80}")
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ AGI")
        print(f"{'='*80}")
        
        print(f"Запрос: {response.user_query}")
        print(f"Архитектура: {response.architecture_used.value}")
        print(f"Время обработки: {response.processing_time:.3f} сек")
        print(f"Уверенность: {response.confidence:.2f}")
        
        print(f"\nСгенерированные команды ({len(response.commands_generated)}):")
        for i, cmd in enumerate(response.commands_generated, 1):
            print(f"  {i}. {cmd}")
        
        print(f"\nОписание состояния:")
        print(f"  {response.state_description}")
        
        print(f"\nИнсайты анализа:")
        for insight in response.analysis_insights:
            print(f"  - {insight}")
        
        print(f"\nРезультаты выполнения:")
        for key, value in response.execution_results.items():
            print(f"  {key}: {value}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Возвращает статус системы"""
        return {
            "total_queries_processed": len(self.processing_history),
            "current_architecture": self.current_architecture.value,
            "average_confidence": sum(r.confidence for r in self.processing_history) / len(self.processing_history) if self.processing_history else 0.0,
            "average_processing_time": sum(r.processing_time for r in self.processing_history) / len(self.processing_history) if self.processing_history else 0.0,
            "last_query": self.processing_history[-1].user_query if self.processing_history else None
        }
    
    def compare_architectures(self, user_query: str) -> Dict[str, AGIResponse]:
        """Сравнивает все архитектуры на одном запросе"""
        print(f"\n{'='*80}")
        print(f"СРАВНЕНИЕ АРХИТЕКТУР")
        print(f"Запрос: {user_query}")
        print(f"{'='*80}")
        
        results = {}
        
        for architecture in AGIArchitecture:
            print(f"\nТестируем архитектуру: {architecture.value}")
            result = self.process_query(user_query, architecture)
            results[architecture.value] = result
            
            print(f"Команд: {len(result.commands_generated)}")
            print(f"Уверенность: {result.confidence:.2f}")
            print(f"Время: {result.processing_time:.3f} сек")
        
        return results

class AGICommandExecutor:
    """Исполнитель команд AGI"""
    
    def __init__(self):
        # Здесь должна быть интеграция с вашим существующим контроллером
        self.execution_history = []
    
    def execute_commands(self, commands: List[str]) -> Dict[str, Any]:
        """Выполняет команды AGI"""
        results = []
        
        for command in commands:
            result = self._execute_single_command(command)
            results.append(result)
        
        return {
            'total_commands': len(commands),
            'successful_commands': len([r for r in results if r['success']]),
            'failed_commands': len([r for r in results if not r['success']]),
            'results': results
        }
    
    def _execute_single_command(self, command: str) -> Dict[str, Any]:
        """Выполняет одну команду"""
        # Упрощенная реализация - в реальности здесь будет ваш контроллер
        return {
            'command': command,
            'success': True,
            'message': f'Команда выполнена: {command}',
            'timestamp': datetime.now()
        }

def test_integrated_agi():
    """Тестирует интегрированную AGI систему"""
    
    agi_system = IntegratedAGISystem()
    
    test_queries = [
        "Создать систему автоматизации с датчиками температуры и влажности",
        "Если температура превышает 25 градусов, то включить кондиционер",
        "Нужно оптимизировать производительность системы и снизить энергопотребление"
    ]
    
    for query in test_queries:
        print(f"\n{'='*100}")
        print(f"ТЕСТИРОВАНИЕ AGI СИСТЕМЫ")
        print(f"Запрос: {query}")
        print(f"{'='*100}")
        
        # Тестируем гибридный подход
        response = agi_system.process_query(query, AGIArchitecture.HYBRID)
        
        print(f"\nСтатус системы:")
        status = agi_system.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

def compare_architectures():
    """Сравнивает все архитектуры"""
    agi_system = IntegratedAGISystem()
    
    test_query = "Создать умную систему управления домом с автоматизацией"
    
    results = agi_system.compare_architectures(test_query)
    
    print(f"\n{'='*80}")
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print(f"{'='*80}")
    
    for arch_name, response in results.items():
        print(f"\n{arch_name.upper()}:")
        print(f"  Команд: {len(response.commands_generated)}")
        print(f"  Уверенность: {response.confidence:.2f}")
        print(f"  Время: {response.processing_time:.3f} сек")
        print(f"  Инсайтов: {len(response.analysis_insights)}")

if __name__ == "__main__":
    print("Запуск тестирования интегрированной AGI системы...")
    test_integrated_agi()
    
    print("\n" + "="*100)
    print("СРАВНЕНИЕ АРХИТЕКТУР")
    print("="*100)
    compare_architectures()