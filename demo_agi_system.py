#!/usr/bin/env python3
"""
Демонстрация интегрированной AGI системы

Этот файл показывает, как использовать различные подходы к проектированию AGI:
1. Система компрессии состояний
2. Иерархические команды
3. Циклическое обучение
4. Интегрированная система
"""

import sys
import time
from typing import Dict, List, Any

def demo_state_compression():
    """Демонстрирует систему компрессии состояний"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ: СИСТЕМА КОМПРЕССИИ СОСТОЯНИЙ")
    print("="*80)
    
    try:
        from state_compression_system import AdaptiveStateController
        
        controller = AdaptiveStateController()
        
        # Тестовые запросы
        test_queries = [
            "Я чувствую себя отлично сегодня",
            "Это логично и правильно",
            "Возможно, это так"
        ]
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            
            # Создаем входной вектор (симуляция)
            import numpy as np
            input_vector = np.random.randn(768)
            
            # Обновляем состояние
            state = controller.update_state(input_vector, query)
            
            # Генерируем команды
            commands = controller.generate_commands_from_state(state)
            
            print(f"  Состояние: {controller.get_state_description()}")
            print(f"  Команды: {commands}")
            
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Убедитесь, что все файлы созданы")

def demo_hierarchical_commands():
    """Демонстрирует иерархические команды"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ: ИЕРАРХИЧЕСКИЕ КОМАНДЫ")
    print("="*80)
    
    try:
        from hierarchical_command_system import ContextAwareController
        
        controller = ContextAwareController()
        
        # Тестовые запросы
        test_queries = [
            "Создать систему автоматизации с 5 датчиками",
            "Если температура высокая, то включить вентиляцию",
            "Установить датчик в позицию x=10, y=20, z=0"
        ]
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            
            # Обрабатываем запрос
            result = controller.process_query(query)
            
            print(f"  Сгенерированные команды:")
            for cmd in result["generated_commands"]:
                print(f"    [{cmd.level.value}] {cmd.command} (уверенность: {cmd.confidence:.2f})")
            
            print(f"  Путь контекста: {' -> '.join(result['context_path'])}")
            print(f"  Общая уверенность: {result['confidence_level']:.2f}")
            
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Убедитесь, что все файлы созданы")

def demo_cyclic_learning():
    """Демонстрирует циклическое обучение"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ: ЦИКЛИЧЕСКОЕ ОБУЧЕНИЕ")
    print("="*80)
    
    try:
        from cyclic_learning_system import CyclicLearningSystem
        
        system = CyclicLearningSystem(max_cycles=3)  # Уменьшаем для демо
        
        # Тестовые запросы
        test_queries = [
            "Создать систему с 5 датчиками температуры",
            "Было 3 яблока, стало 5 яблок",
            "Установить датчик влажности в комнате"
        ]
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            
            # Обрабатываем запрос
            result = system.process_with_cyclic_learning(query)
            
            print(f"  Всего циклов: {result['total_cycles']}")
            print(f"  Общая уверенность: {result['overall_confidence']:.3f}")
            print(f"  Финальный ответ: {result['final_response']}")
            
            # Показываем детали циклов
            for i, cycle in enumerate(result['cycles'][:2]):  # Показываем первые 2 цикла
                print(f"    Цикл {i+1}: {len(cycle.generated_queries)} базовых запросов")
                
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Убедитесь, что все файлы созданы")

def demo_integrated_system():
    """Демонстрирует интегрированную систему"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ: ИНТЕГРИРОВАННАЯ AGI СИСТЕМА")
    print("="*80)
    
    try:
        from integrated_agi_system import IntegratedAGISystem, AGIMode
        
        system = IntegratedAGISystem()
        
        # Тестовые запросы
        test_queries = [
            "Я чувствую себя хорошо сегодня",
            "Создать систему автоматизации с 5 датчиками",
            "Было 3 яблока, стало 5 яблок",
            "Если температура высокая, то включить вентиляцию",
            "Система управления освещением в доме"
        ]
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            
            # Обрабатываем запрос
            result = system.process_query(query)
            
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
        print(f"\nАНАЛИЗ ПАТТЕРНОВ:")
        analysis = system.analyze_query_patterns()
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        # Статус системы
        print(f"\nСТАТУС СИСТЕМЫ:")
        status = system.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
            
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Убедитесь, что все файлы созданы")

def show_system_architecture():
    """Показывает архитектуру системы"""
    print("\n" + "="*80)
    print("АРХИТЕКТУРА AGI СИСТЕМЫ")
    print("="*80)
    
    architecture = """
    ИНТЕГРИРОВАННАЯ AGI СИСТЕМА
    ============================
    
    1. СИСТЕМА КОМПРЕССИИ СОСТОЯНИЙ
       - Сжимает всю логическую цепочку в компактное состояние
       - Описывает эмоции, логику, вероятности, намерения
       - Генерирует команды из состояния без передачи всей истории
    
    2. ИЕРАРХИЧЕСКИЕ КОМАНДЫ
       - Строит контекстное дерево из запроса пользователя
       - Генерирует команды на разных уровнях абстракции
       - От абстрактного к детальному: система -> процесс -> действие -> операция
    
    3. ЦИКЛИЧЕСКОЕ ОБУЧЕНИЕ
       - Генерирует базовые запросы (было X, стало Y)
       - Зацикливает нейросети для построения логических цепочек
       - Адаптируется к контексту пользователя
    
    4. ИНТЕГРИРОВАННАЯ СИСТЕМА
       - Автоматически выбирает оптимальный режим обработки
       - Объединяет результаты всех подсистем
       - Анализирует паттерны и адаптируется
    
    ПРЕИМУЩЕСТВА:
    - Не передает всю логическую цепочку в нейросети
    - Описывает все состояния словами
    - Находит скрытый смысл и закономерности
    - Строит логические цепочки событий
    - Адаптируется к контексту пользователя
    """
    
    print(architecture)

def main():
    """Главная функция демонстрации"""
    print("ДЕМОНСТРАЦИЯ ИНТЕГРИРОВАННОЙ AGI СИСТЕМЫ")
    print("="*80)
    
    # Показываем архитектуру
    show_system_architecture()
    
    # Демонстрируем каждую подсистему
    demo_state_compression()
    demo_hierarchical_commands()
    demo_cyclic_learning()
    demo_integrated_system()
    
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("="*80)
    
    print("\nРЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:")
    print("1. Для эмоциональных/состоятельных запросов - используйте STATE_COMPRESSION")
    print("2. Для системных/структурных запросов - используйте HIERARCHICAL")
    print("3. Для эволюционных/изменяющихся запросов - используйте CYCLIC_LEARNING")
    print("4. Для сложных/неопределенных запросов - используйте HYBRID")
    
    print("\nСЛЕДУЮЩИЕ ШАГИ:")
    print("1. Интегрируйте с вашим существующим контроллером команд")
    print("2. Обучите нейросети на реальных данных")
    print("3. Настройте векторизацию текста")
    print("4. Добавьте обработку ошибок и валидацию")
    print("5. Реализуйте персистентность состояний")

if __name__ == "__main__":
    main()