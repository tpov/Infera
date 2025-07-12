from vectorizer import get_vector
from query_transformer import SimpleQueryTransformer
from command_controller import CommandController
from typing import Dict, List, Any, Optional
import json
import time

class InferaSystem:
    """
    Полная система Infera, объединяющая все три этапа:
    1. Векторизация текста
    2. Генерация команд
    3. Выполнение команд и создание дерева объектов
    """
    
    def __init__(self):
        self.vectorizer = None  # Используем get_vector напрямую
        self.query_transformer = SimpleQueryTransformer()
        self.command_controller = CommandController()
        self.is_trained = False
        
    def train_system(self, training_data: List[Dict[str, Any]], epochs: int = 100):
        """
        Обучает систему на данных
        """
        print("Обучаем систему Infera...")
        self.query_transformer.train(training_data, epochs=epochs)
        self.is_trained = True
        print("Обучение завершено!")
    
    def process_text(self, text: str, max_commands: int = 50) -> Dict[str, Any]:
        """
        Обрабатывает текст через все три этапа
        """
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"ОБРАБОТКА ТЕКСТА: {text}")
        print(f"{'='*80}")
        
        # Этап 1: Векторизация
        print("\n1. ВЕКТОРИЗАЦИЯ")
        print("-" * 40)
        vector = get_vector(text)
        print(f"Создан вектор размерности: {len(vector)}")
        print(f"Первые 5 значений: {vector[:5]}")
        
        # Этап 2: Генерация команд
        print("\n2. ГЕНЕРАЦИЯ КОМАНД")
        print("-" * 40)
        if not self.is_trained:
            print("ВНИМАНИЕ: Система не обучена! Используем базовую генерацию.")
            # Создаем простые команды на основе ключевых слов
            commands = self._generate_basic_commands(text)
        else:
            commands = self.query_transformer.generate_commands(vector, max_commands=max_commands)
        
        print(f"Сгенерированные команды: {commands}")
        
        # Этап 3: Выполнение команд
        print("\n3. ВЫПОЛНЕНИЕ КОМАНД")
        print("-" * 40)
        execution_result = self.command_controller.execute_commands(commands)
        
        # Формируем результат
        result = {
            "input_text": text,
            "vector_dimension": len(vector),
            "vector_sample": vector[:10],
            "generated_commands": commands,
            "execution_success": execution_result.success,
            "execution_message": execution_result.message,
            "created_objects": len(execution_result.created_objects),
            "modified_objects": len(execution_result.modified_objects),
            "deleted_objects": len(execution_result.deleted_objects),
            "contradictions": execution_result.contradictions,
            "warnings": execution_result.warnings,
            "execution_time": execution_result.execution_time,
            "total_processing_time": time.time() - start_time,
            "system_tree": self.command_controller.get_system_tree(),
            "execution_summary": self.command_controller.get_execution_summary()
        }
        
        # Выводим результаты
        self._print_results(result)
        
        return result
    
    def _generate_basic_commands(self, text: str) -> str:
        """
        Генерирует базовые команды на основе ключевых слов
        """
        text_lower = text.lower()
        commands = []
        
        # Простые правила для генерации команд
        if "температур" in text_lower:
            commands.append("set temperature value 25 unit celsius")
        
        if "влажност" in text_lower:
            commands.append("set humidity value 60 unit percent")
        
        if "вентиляц" in text_lower:
            commands.append("create ventilation power value 5 unit kilowatt")
        
        if "робот" in text_lower:
            commands.append("create robot quantity 2 power value 10 unit kilowatt")
        
        if "конвейер" in text_lower:
            commands.append("create conveyor length value 50 unit meter")
        
        if "датчик" in text_lower:
            commands.append("create sensor quantity 5 position distributed")
        
        if "систем" in text_lower:
            commands.append("create system quantity 1 mode value auto")
        
        if "удалить" in text_lower or "убрать" in text_lower:
            commands.append("delete all_objects")
        
        if "включить" in text_lower:
            commands.append("activate all_objects")
        
        if "выключить" in text_lower:
            commands.append("deactivate all_objects")
        
        if not commands:
            commands.append("create system quantity 1 status value active")
        
        return f"[{', '.join(commands)}]"
    
    def _print_results(self, result: Dict[str, Any]):
        """
        Выводит результаты обработки
        """
        print(f"\n{'='*80}")
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ")
        print(f"{'='*80}")
        
        print(f"\nВходной текст: {result['input_text']}")
        print(f"Размерность вектора: {result['vector_dimension']}")
        print(f"Сгенерированные команды: {result['generated_commands']}")
        
        print(f"\nРЕЗУЛЬТАТ ВЫПОЛНЕНИЯ:")
        print(f"  Успех: {result['execution_success']}")
        print(f"  Сообщение: {result['execution_message']}")
        print(f"  Создано объектов: {result['created_objects']}")
        print(f"  Изменено объектов: {result['modified_objects']}")
        print(f"  Удалено объектов: {result['deleted_objects']}")
        print(f"  Время выполнения: {result['execution_time']:.3f} сек")
        print(f"  Общее время обработки: {result['total_processing_time']:.3f} сек")
        
        if result['contradictions']:
            print(f"\nПРОТИВОРЕЧИЯ:")
            for contradiction in result['contradictions']:
                print(f"  - {contradiction}")
        
        if result['warnings']:
            print(f"\nПРЕДУПРЕЖДЕНИЯ:")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        print(f"\nДЕРЕВО ОБЪЕКТОВ:")
        tree = result['system_tree']
        print(f"  Всего объектов: {tree['total_objects']}")
        print(f"  Типы объектов: {tree['object_types']}")
        
        if tree['objects']:
            print(f"  Детали объектов:")
            for obj_id, obj_data in tree['objects'].items():
                print(f"    {obj_id}: {obj_data['name']} ({obj_data['type']}) - {obj_data['state']}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Возвращает статус системы
        """
        return {
            "is_trained": self.is_trained,
            "total_objects": len(self.command_controller.system_state.objects),
            "execution_history_length": len(self.command_controller.system_state.execution_history),
            "available_tokens": len(self.query_transformer.get_available_tokens()),
            "system_tree": self.command_controller.get_system_tree()
        }
    
    def reset_system(self):
        """
        Сбрасывает состояние системы
        """
        self.command_controller = CommandController()
        print("Состояние системы сброшено")

def create_sample_training_data() -> List[Dict[str, Any]]:
    """
    Создает примеры данных для обучения
    """
    from vectorizer import get_vector
    
    sample_data = [
        {
            "text": "Создать систему автоматизации с 5 датчиками температуры и 3 роботами",
            "commands": "[create temperature_sensor quantity 5 position distributed, create robot quantity 3 power value 10 unit kilowatt, create automation_system quantity 1 mode value auto]"
        },
        {
            "text": "Установить температуру 25 градусов и влажность 60%",
            "commands": "[set temperature value 25 unit celsius, set humidity value 60 unit percent]"
        },
        {
            "text": "Включить вентиляцию на полную мощность и выключить отопление",
            "commands": "[set ventilation power value full, set heating state off]"
        },
        {
            "text": "Создать конвейер длиной 100 метров с 2 моторами",
            "commands": "[create conveyor length value 100 unit meter, create motor quantity 2 power value 5 unit kilowatt]"
        },
        {
            "text": "Настроить систему безопасности с 10 камерами",
            "commands": "[configure security_system quantity 10 cameras position strategic]"
        },
        {
            "text": "Удалить все настройки и создать новую систему",
            "commands": "[delete all_settings, create new_system quantity 1 mode value auto]"
        },
        {
            "text": "Активировать все роботы и деактивировать датчики",
            "commands": "[activate robot, deactivate sensor]"
        },
        {
            "text": "Создать промышленную сеть с серверами и базами данных",
            "commands": "[create industrial_network quantity 1, create server quantity 3, create database quantity 2]"
        }
    ]
    
    # Векторизуем тексты
    for item in sample_data:
        item['vector'] = get_vector(item['text'])
        del item['text']
    
    return sample_data

if __name__ == "__main__":
    # Создаем систему
    system = InferaSystem()
    
    # Обучаем систему
    print("ОБУЧЕНИЕ СИСТЕМЫ INFERA")
    print("=" * 80)
    training_data = create_sample_training_data()
    system.train_system(training_data, epochs=50)
    
    # Тестируем систему
    print("\n\nТЕСТИРОВАНИЕ СИСТЕМЫ INFERA")
    print("=" * 80)
    
    test_cases = [
        "Создать автоматизированную систему управления заводом с множественными конвейерами, роботами и датчиками",
        "Установить температуру 30 градусов Цельсия и влажность 70% в помещении",
        "Включить все роботы на полную мощность и активировать систему безопасности",
        "Создать промышленную сеть с 5 серверами и 3 базами данных",
        "Настроить систему вентиляции с 10 вентиляторами мощностью 5 кВт каждый",
        "Удалить все старые настройки и создать новую систему автоматизации"
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n\nТЕСТ {i}: {test_text}")
        result = system.process_text(test_text)
        
        # Показываем статус системы
        status = system.get_system_status()
        print(f"\nСтатус системы после теста {i}:")
        print(f"  Всего объектов: {status['total_objects']}")
        print(f"  История выполнения: {status['execution_history_length']} записей")
    
    # Финальный статус
    print(f"\n\n{'='*80}")
    print("ФИНАЛЬНЫЙ СТАТУС СИСТЕМЫ")
    print(f"{'='*80}")
    final_status = system.get_system_status()
    print(json.dumps(final_status, indent=2, ensure_ascii=False)) 