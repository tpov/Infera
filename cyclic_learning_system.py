from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import random
import re
from enum import Enum

class QueryType(Enum):
    BASIC = "basic"           # Базовые запросы (было X, стало Y)
    ANALYTICAL = "analytical" # Аналитические запросы
    PREDICTIVE = "predictive" # Прогностические запросы
    CREATIVE = "creative"     # Творческие запросы

@dataclass
class BasicQuery:
    """Базовый запрос для обучения"""
    query_type: QueryType
    original_state: str
    final_state: str
    change_description: str
    confidence: float
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class LearningCycle:
    """Цикл обучения"""
    cycle_id: int
    input_query: str
    generated_queries: List[BasicQuery]
    controller_commands: List[str]
    controller_results: Dict[str, Any]
    neural_response: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

class BasicQueryGenerator:
    """Генератор базовых запросов"""
    
    def __init__(self):
        self.query_templates = self._load_query_templates()
        self.state_vocabulary = self._load_state_vocabulary()
        self.change_patterns = self._load_change_patterns()
    
    def generate_basic_queries(self, user_query: str, max_queries: int = 5) -> List[BasicQuery]:
        """Генерирует базовые запросы из пользовательского запроса"""
        queries = []
        
        # Анализируем запрос пользователя
        extracted_info = self._extract_query_info(user_query)
        
        # Генерируем базовые запросы
        for i in range(max_queries):
            query = self._generate_single_query(extracted_info, i)
            if query:
                queries.append(query)
        
        return queries
    
    def _extract_query_info(self, user_query: str) -> Dict[str, Any]:
        """Извлекает информацию из запроса пользователя"""
        info = {
            'objects': [],
            'actions': [],
            'quantities': [],
            'properties': [],
            'relations': []
        }
        
        query_lower = user_query.lower()
        
        # Извлекаем объекты
        object_patterns = [
            r'(\d+)\s+(\w+)',  # 5 датчиков
            r'(\w+)\s+(\w+)',  # система автоматизации
            r'(\w+)',           # отдельные слова
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if isinstance(match, tuple):
                    info['objects'].extend(match)
                else:
                    info['objects'].append(match)
        
        # Извлекаем действия
        action_keywords = ['создать', 'удалить', 'изменить', 'установить', 'включить', 'выключить']
        for action in action_keywords:
            if action in query_lower:
                info['actions'].append(action)
        
        # Извлекаем количества
        quantity_pattern = r'(\d+)'
        quantities = re.findall(quantity_pattern, query_lower)
        info['quantities'] = [int(q) for q in quantities]
        
        return info
    
    def _generate_single_query(self, extracted_info: Dict[str, Any], index: int) -> Optional[BasicQuery]:
        """Генерирует один базовый запрос"""
        
        # Выбираем случайный шаблон
        template = random.choice(self.query_templates)
        
        # Заполняем шаблон данными
        if template['type'] == 'state_change':
            original_state = self._generate_original_state(extracted_info)
            final_state = self._generate_final_state(extracted_info)
            change_description = f"Изменение состояния: {original_state} -> {final_state}"
            
            return BasicQuery(
                query_type=QueryType.BASIC,
                original_state=original_state,
                final_state=final_state,
                change_description=change_description,
                confidence=0.8
            )
        
        elif template['type'] == 'quantity_change':
            if extracted_info['quantities']:
                original_qty = random.choice(extracted_info['quantities'])
                final_qty = original_qty + random.randint(-2, 2)
                if final_qty < 0:
                    final_qty = 0
                
                object_name = random.choice(extracted_info['objects']) if extracted_info['objects'] else 'объект'
                
                return BasicQuery(
                    query_type=QueryType.BASIC,
                    original_state=f"было {original_qty} {object_name}",
                    final_state=f"стало {final_qty} {object_name}",
                    change_description=f"Изменение количества {object_name}: {original_qty} -> {final_qty}",
                    confidence=0.9
                )
        
        return None
    
    def _generate_original_state(self, info: Dict[str, Any]) -> str:
        """Генерирует исходное состояние"""
        states = [
            "система неактивна",
            "датчики отключены",
            "вентиляция выключена",
            "температура нормальная",
            "влажность в норме",
            "освещение выключено",
            "безопасность неактивна"
        ]
        
        if info['objects']:
            object_name = random.choice(info['objects'])
            return f"{object_name} неактивен"
        
        return random.choice(states)
    
    def _generate_final_state(self, info: Dict[str, Any]) -> str:
        """Генерирует конечное состояние"""
        states = [
            "система активна",
            "датчики включены",
            "вентиляция работает",
            "температура повышена",
            "влажность изменена",
            "освещение включено",
            "безопасность активна"
        ]
        
        if info['objects']:
            object_name = random.choice(info['objects'])
            return f"{object_name} активен"
        
        return random.choice(states)
    
    def _load_query_templates(self) -> List[Dict[str, Any]]:
        """Загружает шаблоны запросов"""
        return [
            {
                'type': 'state_change',
                'description': 'Изменение состояния объекта'
            },
            {
                'type': 'quantity_change',
                'description': 'Изменение количества объектов'
            },
            {
                'type': 'property_change',
                'description': 'Изменение свойств объекта'
            }
        ]
    
    def _load_state_vocabulary(self) -> Dict[str, List[str]]:
        """Загружает словарь состояний"""
        return {
            'objects': ['датчик', 'система', 'вентиляция', 'освещение', 'безопасность', 'контроллер'],
            'states': ['активен', 'неактивен', 'включен', 'выключен', 'работает', 'остановлен'],
            'properties': ['температура', 'влажность', 'давление', 'скорость', 'мощность', 'напряжение']
        }
    
    def _load_change_patterns(self) -> List[str]:
        """Загружает паттерны изменений"""
        return [
            "было {original}, стало {final}",
            "изменилось с {original} на {final}",
            "переход от {original} к {final}",
            "эволюция от {original} до {final}"
        ]

class CyclicLearningSystem:
    """Система циклического обучения"""
    
    def __init__(self, max_cycles: int = 10):
        self.max_cycles = max_cycles
        self.query_generator = BasicQueryGenerator()
        self.learning_history = []
        self.current_cycle = 0
        
    def process_with_cyclic_learning(self, user_query: str) -> Dict[str, Any]:
        """Обрабатывает запрос с циклическим обучением"""
        
        print(f"Начинаем циклическое обучение для запроса: {user_query}")
        print(f"Максимальное количество циклов: {self.max_cycles}")
        
        cycles = []
        current_input = user_query
        
        for cycle in range(self.max_cycles):
            print(f"\n--- Цикл {cycle + 1} ---")
            
            # Генерируем базовые запросы
            basic_queries = self.query_generator.generate_basic_queries(current_input)
            print(f"Сгенерировано базовых запросов: {len(basic_queries)}")
            
            # Выполняем команды контроллера (симуляция)
            controller_commands = self._generate_controller_commands(basic_queries)
            controller_results = self._execute_controller_commands(controller_commands)
            
            # Генерируем ответ нейросети
            neural_response = self._generate_neural_response(basic_queries, controller_results)
            
            # Создаем цикл обучения
            learning_cycle = LearningCycle(
                cycle_id=cycle + 1,
                input_query=current_input,
                generated_queries=basic_queries,
                controller_commands=controller_commands,
                controller_results=controller_results,
                neural_response=neural_response,
                confidence=self._calculate_cycle_confidence(basic_queries, controller_results)
            )
            
            cycles.append(learning_cycle)
            
            # Проверяем условия остановки
            if self._should_stop_learning(learning_cycle):
                print(f"Останавливаем обучение на цикле {cycle + 1}")
                break
            
            # Подготавливаем вход для следующего цикла
            current_input = self._prepare_next_cycle_input(learning_cycle)
            print(f"Подготовлен вход для следующего цикла: {current_input[:100]}...")
        
        # Сохраняем историю
        self.learning_history.extend(cycles)
        
        return {
            "user_query": user_query,
            "total_cycles": len(cycles),
            "cycles": cycles,
            "final_response": cycles[-1].neural_response if cycles else "Нет ответа",
            "overall_confidence": self._calculate_overall_confidence(cycles),
            "learning_summary": self._generate_learning_summary(cycles)
        }
    
    def _generate_controller_commands(self, basic_queries: List[BasicQuery]) -> List[str]:
        """Генерирует команды контроллера из базовых запросов"""
        commands = []
        
        for query in basic_queries:
            if query.query_type == QueryType.BASIC:
                # Преобразуем базовый запрос в команду
                command = self._basic_query_to_command(query)
                if command:
                    commands.append(command)
        
        return commands
    
    def _basic_query_to_command(self, query: BasicQuery) -> Optional[str]:
        """Преобразует базовый запрос в команду контроллера"""
        
        # Анализируем изменение состояния
        if "было" in query.original_state and "стало" in query.final_state:
            # Извлекаем объект и действие
            original_parts = query.original_state.split()
            final_parts = query.final_state.split()
            
            if len(original_parts) >= 2 and len(final_parts) >= 2:
                original_qty = original_parts[0] if original_parts[0].isdigit() else "1"
                final_qty = final_parts[0] if final_parts[0].isdigit() else "1"
                object_name = original_parts[1] if original_parts[0].isdigit() else original_parts[0]
                
                if original_qty != final_qty:
                    if int(final_qty) > int(original_qty):
                        return f"create {object_name} quantity {int(final_qty) - int(original_qty)}"
                    else:
                        return f"delete {object_name} quantity {int(original_qty) - int(final_qty)}"
        
        # Если не удалось разобрать, создаем общую команду
        return f"create system quantity 1"
    
    def _execute_controller_commands(self, commands: List[str]) -> Dict[str, Any]:
        """Выполняет команды контроллера (симуляция)"""
        results = {
            "success": True,
            "created_objects": len(commands),
            "modified_objects": 0,
            "deleted_objects": 0,
            "execution_time": 0.1,
            "warnings": [],
            "errors": []
        }
        
        for command in commands:
            if "delete" in command:
                results["deleted_objects"] += 1
            elif "create" in command:
                results["created_objects"] += 1
            elif "modify" in command:
                results["modified_objects"] += 1
        
        return results
    
    def _generate_neural_response(self, basic_queries: List[BasicQuery], controller_results: Dict[str, Any]) -> str:
        """Генерирует ответ нейросети"""
        
        if not basic_queries:
            return "Не удалось сгенерировать базовые запросы"
        
        # Анализируем результаты
        response_parts = []
        
        # Добавляем информацию о базовых запросах
        query_summary = []
        for query in basic_queries:
            query_summary.append(f"{query.original_state} -> {query.final_state}")
        
        response_parts.append(f"Базовые изменения: {'; '.join(query_summary)}")
        
        # Добавляем информацию о выполнении команд
        if controller_results["created_objects"] > 0:
            response_parts.append(f"Создано объектов: {controller_results['created_objects']}")
        
        if controller_results["modified_objects"] > 0:
            response_parts.append(f"Изменено объектов: {controller_results['modified_objects']}")
        
        if controller_results["deleted_objects"] > 0:
            response_parts.append(f"Удалено объектов: {controller_results['deleted_objects']}")
        
        # Формируем итоговый ответ
        response = ". ".join(response_parts)
        
        return response
    
    def _calculate_cycle_confidence(self, basic_queries: List[BasicQuery], controller_results: Dict[str, Any]) -> float:
        """Вычисляет уверенность цикла"""
        if not basic_queries:
            return 0.0
        
        # Средняя уверенность базовых запросов
        query_confidence = sum(q.confidence for q in basic_queries) / len(basic_queries)
        
        # Уверенность выполнения команд
        execution_confidence = 1.0 if controller_results["success"] else 0.5
        
        # Общая уверенность
        return (query_confidence + execution_confidence) / 2
    
    def _should_stop_learning(self, cycle: LearningCycle) -> bool:
        """Определяет, нужно ли остановить обучение"""
        
        # Останавливаем, если уверенность высокая
        if cycle.confidence > 0.8:
            return True
        
        # Останавливаем, если нет базовых запросов
        if not cycle.generated_queries:
            return True
        
        # Останавливаем, если команды не выполняются
        if not cycle.controller_results["success"]:
            return True
        
        # Останавливаем после минимального количества циклов
        if cycle.cycle_id >= 5:
            return True
        
        return False
    
    def _prepare_next_cycle_input(self, cycle: LearningCycle) -> str:
        """Подготавливает вход для следующего цикла"""
        
        # Используем ответ нейросети как вход для следующего цикла
        return cycle.neural_response
    
    def _calculate_overall_confidence(self, cycles: List[LearningCycle]) -> float:
        """Вычисляет общую уверенность"""
        if not cycles:
            return 0.0
        
        total_confidence = sum(cycle.confidence for cycle in cycles)
        return total_confidence / len(cycles)
    
    def _generate_learning_summary(self, cycles: List[LearningCycle]) -> Dict[str, Any]:
        """Генерирует сводку обучения"""
        if not cycles:
            return {"message": "Нет данных для сводки"}
        
        total_queries = sum(len(cycle.generated_queries) for cycle in cycles)
        total_commands = sum(len(cycle.controller_commands) for cycle in cycles)
        avg_confidence = self._calculate_overall_confidence(cycles)
        
        return {
            "total_cycles": len(cycles),
            "total_basic_queries": total_queries,
            "total_controller_commands": total_commands,
            "average_confidence": avg_confidence,
            "final_cycle_confidence": cycles[-1].confidence if cycles else 0.0
        }

def test_cyclic_learning():
    """Тестирует систему циклического обучения"""
    system = CyclicLearningSystem(max_cycles=5)
    
    test_queries = [
        "Создать систему с 5 датчиками температуры",
        "Если температура выше 25 градусов, включить вентиляцию",
        "Установить датчик влажности в комнате"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"ТЕСТИРОВАНИЕ: {query}")
        print(f"{'='*80}")
        
        result = system.process_with_cyclic_learning(query)
        
        print(f"\nРЕЗУЛЬТАТЫ:")
        print(f"  Всего циклов: {result['total_cycles']}")
        print(f"  Общая уверенность: {result['overall_confidence']:.3f}")
        print(f"  Финальный ответ: {result['final_response']}")
        
        print(f"\nСВОДКА ОБУЧЕНИЯ:")
        summary = result['learning_summary']
        for key, value in summary.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_cyclic_learning()