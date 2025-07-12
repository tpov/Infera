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
# ФИНАЛЬНАЯ AGI СИСТЕМА
# ============================================================================

class FinalAGISystem:
    """Финальная интегрированная AGI система"""
    
    def __init__(self):
        # Импортируем все компоненты
        from universal_command_system import UniversalController
        from cyclic_learning_system import CyclicLearningSystem
        
        self.controller = UniversalController()
        self.learning_system = CyclicLearningSystem(max_cycles=3)
        
        # Состояние системы
        self.conversation_history = []
        self.learning_mode = False
        self.context_depth = 0
        self.max_context_depth = 5
        
    def process_input(self, user_input: str, enable_learning: bool = True) -> Dict[str, Any]:
        """Обрабатывает любой пользовательский ввод с возможностью обучения"""
        
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"AGI СИСТЕМА: {user_input}")
        print(f"{'='*80}")
        
        # 1. Анализируем контекст и глубину
        context_analysis = self._analyze_context_depth(user_input)
        
        # 2. Решаем, нужно ли обучение
        if enable_learning and self._should_learn(user_input, context_analysis):
            print("Включаем режим обучения...")
            self.learning_mode = True
            learning_result = self.learning_system.learn_from_input(user_input)
            
            # Используем результаты обучения для финальной обработки
            final_result = self.controller.process_input(user_input)
            
            # Объединяем результаты
            result = {
                "mode": "learning",
                "original_input": user_input,
                "learning_result": learning_result,
                "final_result": final_result,
                "context_depth": context_analysis['depth'],
                "processing_time": time.time() - start_time,
                "system_state": self.get_system_state()
            }
        else:
            print("Стандартная обработка...")
            self.learning_mode = False
            result = self.controller.process_input(user_input)
            result.update({
                "mode": "standard",
                "context_depth": context_analysis['depth'],
                "processing_time": time.time() - start_time,
                "system_state": self.get_system_state()
            })
        
        # 3. Обновляем историю разговора
        self._update_conversation_history(user_input, result)
        
        # 4. Выводим результаты
        self._print_results(result)
        
        return result
    
    def _analyze_context_depth(self, user_input: str) -> Dict[str, Any]:
        """Анализирует глубину контекста"""
        
        # Простые эвристики для определения глубины
        depth_indicators = {
            'simple': ['да', 'нет', 'хорошо', 'плохо'],
            'medium': ['почему', 'как', 'что', 'когда'],
            'complex': ['если', 'то', 'потому что', 'следовательно', 'однако'],
            'very_complex': ['на самом деле', 'по сути', 'в действительности', 'между строк']
        }
        
        depth = 1  # Базовая глубина
        
        for level, indicators in depth_indicators.items():
            if any(indicator in user_input.lower() for indicator in indicators):
                if level == 'simple':
                    depth = 1
                elif level == 'medium':
                    depth = 2
                elif level == 'complex':
                    depth = 3
                elif level == 'very_complex':
                    depth = 4
        
        # Учитываем длину и сложность
        words = user_input.split()
        if len(words) > 20:
            depth = min(5, depth + 1)
        
        return {
            'depth': depth,
            'word_count': len(words),
            'complexity': min(1.0, len(words) / 20.0)
        }
    
    def _should_learn(self, user_input: str, context_analysis: Dict[str, Any]) -> bool:
        """Определяет, нужно ли включить режим обучения"""
        
        # Учимся на сложных запросах
        if context_analysis['depth'] >= 3:
            return True
        
        # Учимся на новых типах запросов
        if not self._is_familiar_pattern(user_input):
            return True
        
        # Учимся на эмоциональных запросах
        emotional_words = ['грустно', 'радость', 'любовь', 'страх', 'злость', 'счастье']
        if any(word in user_input.lower() for word in emotional_words):
            return True
        
        # Учимся на вопросах
        question_words = ['что', 'как', 'почему', 'когда', 'где', 'кто', 'зачем']
        if any(word in user_input.lower() for word in question_words):
            return True
        
        return False
    
    def _is_familiar_pattern(self, user_input: str) -> bool:
        """Проверяет, знаком ли паттерн"""
        
        # Простая проверка по истории
        for entry in self.conversation_history[-10:]:  # Последние 10 записей
            if self._similarity(user_input, entry['input']) > 0.7:
                return True
        
        return False
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Вычисляет простую схожесть текстов"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _update_conversation_history(self, user_input: str, result: Dict[str, Any]):
        """Обновляет историю разговора"""
        
        entry = {
            'input': user_input,
            'timestamp': datetime.now(),
            'mode': result.get('mode', 'unknown'),
            'context_depth': result.get('context_depth', 1),
            'response': result.get('response', ''),
            'commands_count': len(result.get('universal_commands', [])),
            'success': result.get('execution_result', {}).get('success', False)
        }
        
        self.conversation_history.append(entry)
        
        # Ограничиваем историю
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-50:]
    
    def _print_results(self, result: Dict[str, Any]):
        """Выводит результаты обработки"""
        
        print(f"\nРЕЗУЛЬТАТЫ ОБРАБОТКИ:")
        print(f"  Режим: {result.get('mode', 'unknown')}")
        print(f"  Глубина контекста: {result.get('context_depth', 1)}")
        print(f"  Ответ: {result.get('response', 'Нет ответа')}")
        print(f"  Время обработки: {result.get('processing_time', 0):.3f} сек")
        
        if result.get('mode') == 'learning':
            learning_stats = result.get('learning_result', {}).get('learning_statistics', {})
            print(f"  Циклов обучения: {learning_stats.get('total_cycles', 0)}")
            print(f"  Средняя оценка: {learning_stats.get('average_learning_score', 0):.3f}")
        
        # Выводим команды
        commands = result.get('universal_commands', [])
        if commands:
            print(f"  Универсальных команд: {len(commands)}")
            for i, cmd in enumerate(commands[:3]):  # Показываем первые 3
                print(f"    {i+1}. {cmd.get('command_type', 'unknown')}: {cmd.get('action', 'unknown')}")
    
    def get_system_state(self) -> Dict[str, Any]:
        """Возвращает состояние системы"""
        return {
            "conversation_history_length": len(self.conversation_history),
            "learning_mode": self.learning_mode,
            "context_depth": self.context_depth,
            "controller_state": self.controller.get_state_summary(),
            "learning_statistics": self.learning_system._get_learning_statistics() if hasattr(self.learning_system, '_get_learning_statistics') else {}
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Возвращает сводку разговора"""
        if not self.conversation_history:
            return {"message": "История разговора пуста"}
        
        recent_entries = self.conversation_history[-10:]
        
        return {
            "total_entries": len(self.conversation_history),
            "recent_entries": len(recent_entries),
            "learning_entries": len([e for e in recent_entries if e.get('mode') == 'learning']),
            "average_context_depth": sum(e.get('context_depth', 1) for e in recent_entries) / len(recent_entries),
            "success_rate": len([e for e in recent_entries if e.get('success', False)]) / len(recent_entries),
            "last_input": recent_entries[-1]['input'] if recent_entries else None
        }

# ============================================================================
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# ============================================================================

def interactive_mode():
    """Интерактивный режим для тестирования AGI системы"""
    
    print("Добро пожаловать в AGI систему!")
    print("Введите любой текст, и система обработает его.")
    print("Для выхода введите 'выход' или 'exit'")
    print("Для получения статистики введите 'статистика'")
    print("Для включения/выключения обучения введите 'обучение'")
    print("-" * 80)
    
    system = FinalAGISystem()
    learning_enabled = True
    
    while True:
        try:
            user_input = input("\nВы: ").strip()
            
            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("До свидания!")
                break
            
            elif user_input.lower() == 'статистика':
                summary = system.get_conversation_summary()
                print(f"\nСТАТИСТИКА РАЗГОВОРА:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
                continue
            
            elif user_input.lower() == 'обучение':
                learning_enabled = not learning_enabled
                status = "включено" if learning_enabled else "выключено"
                print(f"Режим обучения {status}")
                continue
            
            elif not user_input:
                continue
            
            # Обрабатываем ввод
            result = system.process_input(user_input, enable_learning=learning_enabled)
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

def test_final_agi_system():
    """Тестирует финальную AGI систему"""
    print("Тестирование финальной AGI системы...")
    
    system = FinalAGISystem()
    
    # Тестовые запросы разной сложности
    test_inputs = [
        # Простые запросы
        "Привет",
        "Да",
        "Нет",
        
        # Средние запросы
        "Что такое AGI?",
        "Как работает система?",
        "Почему небо голубое?",
        
        # Сложные запросы
        "Если я создам робота, то он сможет думать, потому что у него будет искусственный интеллект",
        "На самом деле, все сводится к пониманию контекста и скрытого смысла",
        "Между строк я вижу, что ты хочешь сказать больше, чем говоришь",
        
        # Эмоциональные запросы
        "Мне грустно сегодня",
        "Я чувствую радость от общения с тобой",
        "Люблю музыку и искусство",
        
        # Команды
        "Создай систему автоматизации",
        "Анализируй данные",
        "Сравни разные подходы"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{'='*80}")
        print(f"ТЕСТ {i}: {user_input}")
        print(f"{'='*80}")
        
        # Тестируем с обучением
        result_with_learning = system.process_input(user_input, enable_learning=True)
        
        # Тестируем без обучения
        result_without_learning = system.process_input(user_input, enable_learning=False)
        
        print(f"\nСРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
        print(f"  С обучением: {result_with_learning.get('mode', 'unknown')}")
        print(f"  Без обучения: {result_without_learning.get('mode', 'unknown')}")
        
        # Выводим статистику
        if i % 5 == 0:
            summary = system.get_conversation_summary()
            print(f"\nПРОМЕЖУТОЧНАЯ СТАТИСТИКА:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
    
    # Финальная статистика
    print(f"\n{'='*80}")
    print("ФИНАЛЬНАЯ СТАТИСТИКА")
    print(f"{'='*80}")
    
    final_summary = system.get_conversation_summary()
    for key, value in final_summary.items():
        print(f"{key}: {value}")
    
    system_state = system.get_system_state()
    print(f"\nСОСТОЯНИЕ СИСТЕМЫ:")
    for key, value in system_state.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        test_final_agi_system()