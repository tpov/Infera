import torch
from typing import Dict, Any, Optional, List
from vectorizer import Vectorizer
from command_controller import CommandController
from intermediate_network import IntermediateNetworkWrapper
from final_generative_network import FinalGenerativeNetworkWrapper

class ThinkingPipeline:
    """
    Пайплайн с режимом "думания"
    Результаты подаются обратно на вход первого этапа до 20 циклов
    """
    
    def __init__(self,
                 vectorizer_model_path: Optional[str] = None,
                 intermediate_model_path: Optional[str] = None,
                 final_model_path: Optional[str] = None,
                 max_thinking_cycles: int = 20):
        
        # Инициализируем компоненты
        self.vectorizer = Vectorizer(vectorizer_model_path)
        self.controller = CommandController()
        self.intermediate_network = IntermediateNetworkWrapper(intermediate_model_path)
        self.final_network = FinalGenerativeNetworkWrapper(final_model_path)
        
        self.max_thinking_cycles = max_thinking_cycles
        
    def process_with_thinking(self, user_query: str) -> Dict[str, Any]:
        """
        Обрабатывает запрос в режиме "думания"
        
        Args:
            user_query: Исходный запрос пользователя
        
        Returns:
            Dict с результатами всех циклов и финальным ответом
        """
        
        results = {
            "user_query": user_query,
            "thinking_cycles": [],
            "final_response": None,
            "success": False,
            "error": None
        }
        
        try:
            # Начинаем с исходного запроса
            current_input = user_query
            cycle_results = []
            
            print(f"Начинаем режим 'думания' с максимальным количеством циклов: {self.max_thinking_cycles}")
            
            for cycle in range(self.max_thinking_cycles):
                print(f"\n--- Цикл {cycle + 1} ---")
                
                # Этап 1: Векторизация
                print("Этап 1: Векторизация...")
                vector = self.vectorizer.vectorize(current_input)
                print(f"Вектор создан, размер: {vector.shape}")
                
                # Этап 2: Генерация команд (простая логика)
                print("Этап 2: Генерация команд...")
                commands = self._generate_commands_from_vector(vector)
                print(f"Сгенерированы команды: {commands}")
                
                # Этап 3: Выполнение команд контроллером
                print("Этап 3: Выполнение команд...")
                controller_result = self.controller.execute_commands(commands)
                print(f"Контроллер выполнен: {controller_result.success}")
                
                # Этап 4: Промежуточная сеть
                print("Этап 4: Промежуточная сеть...")
                try:
                    intermediate_vector = self.intermediate_network.process(current_input, {
                        "success": controller_result.success,
                        "message": controller_result.message,
                        "created_objects": controller_result.created_objects,
                        "modified_objects": controller_result.modified_objects,
                        "computed_values": controller_result.computed_values,
                        "contradictions": controller_result.contradictions,
                        "warnings": controller_result.warnings
                    })
                    print(f"Промежуточный вектор создан: {intermediate_vector.shape}")
                except Exception as e:
                    print(f"Ошибка в промежуточной сети: {e}")
                    intermediate_vector = torch.randn(1, 768)
                
                # Этап 5: Финальная генеративная сеть
                print("Этап 5: Генерация ответа...")
                try:
                    response = self.final_network.generate_response(intermediate_vector)
                    print(f"Сгенерирован ответ: {response}")
                except Exception as e:
                    print(f"Ошибка в финальной сети: {e}")
                    response = f"Цикл {cycle + 1}: Ошибка генерации"
                
                # Сохраняем результаты цикла
                cycle_result = {
                    "cycle": cycle + 1,
                    "input": current_input,
                    "vector": vector,
                    "commands": commands,
                    "controller_result": {
                        "success": controller_result.success,
                        "message": controller_result.message,
                        "created_objects": len(controller_result.created_objects),
                        "modified_objects": len(controller_result.modified_objects),
                        "computed_values": controller_result.computed_values,
                        "contradictions": controller_result.contradictions,
                        "warnings": controller_result.warnings
                    },
                    "intermediate_vector": intermediate_vector,
                    "response": response
                }
                
                cycle_results.append(cycle_result)
                
                # Проверяем, нужно ли продолжать "думание"
                if self._should_stop_thinking(cycle_result):
                    print(f"Останавливаем 'думание' на цикле {cycle + 1}")
                    break
                
                # Подготавливаем вход для следующего цикла
                current_input = self._prepare_next_cycle_input(cycle_result)
                print(f"Подготовлен вход для следующего цикла: {current_input[:100]}...")
            
            # Сохраняем все результаты
            results["thinking_cycles"] = cycle_results
            results["final_response"] = cycle_results[-1]["response"] if cycle_results else "Нет ответа"
            results["success"] = True
            
        except Exception as e:
            results["error"] = str(e)
            print(f"Ошибка в пайплайне: {e}")
        
        return results
    
    def _generate_commands_from_vector(self, vector: torch.Tensor) -> str:
        """
        Простая логика генерации команд из вектора
        """
        vector_norm = torch.norm(vector).item()
        
        if vector_norm > 0.5:
            return "create robot age 25 cost 100"
        else:
            return "create sensor name test_sensor"
    
    def _should_stop_thinking(self, cycle_result: Dict[str, Any]) -> bool:
        """
        Определяет, нужно ли остановить "думание"
        """
        # Останавливаем, если контроллер успешно выполнил команды
        if cycle_result["controller_result"]["success"]:
            return True
        
        # Останавливаем, если нет ошибок
        if not cycle_result["controller_result"]["contradictions"] and not cycle_result["controller_result"]["warnings"]:
            return True
        
        # Останавливаем, если созданы объекты с ошибками
        if cycle_result["controller_result"]["created_objects"] > 0:
            return True
        
        return False
    
    def _prepare_next_cycle_input(self, cycle_result: Dict[str, Any]) -> str:
        """
        Подготавливает вход для следующего цикла
        """
        original_query = cycle_result["input"]
        response = cycle_result["response"]
        controller_result = cycle_result["controller_result"]
        
        # Объединяем исходный запрос с результатами
        next_input = f"{original_query} | Результат: {response}"
        
        # Добавляем информацию о созданных объектах
        if controller_result["created_objects"] > 0:
            next_input += f" | Создано объектов: {controller_result['created_objects']}"
        
        # Добавляем информацию о противоречиях
        if controller_result["contradictions"]:
            next_input += f" | Противоречия: {', '.join(controller_result['contradictions'])}"
        
        # Добавляем информацию о предупреждениях
        if controller_result["warnings"]:
            next_input += f" | Предупреждения: {', '.join(controller_result['warnings'])}"
        
        return next_input
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Возвращает статус системы
        """
        return {
            "total_objects": len(self.controller.system_state.objects),
            "object_types": self.controller.get_system_tree()["object_types"],
            "execution_history": len(self.controller.system_state.execution_history)
        }

def test_thinking_pipeline():
    """
    Тестирует пайплайн с режимом "думания"
    """
    print("Тестирование пайплайна с режимом 'думания'...")
    
    # Создаем пайплайн
    pipeline = ThinkingPipeline(max_thinking_cycles=5)  # Уменьшаем для теста
    
    # Тестовые запросы
    test_queries = [
        "Создай робота с возрастом -5 и стоимостью 100",
        "Создай сенсор с именем it1",
        "Создай устройство с количеством 0"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Тест {i}: {query}")
        print(f"{'='*60}")
        
        result = pipeline.process_with_thinking(query)
        
        print(f"\nУспех: {result['success']}")
        if result['error']:
            print(f"Ошибка: {result['error']}")
        
        print(f"Количество циклов: {len(result['thinking_cycles'])}")
        print(f"Финальный ответ: {result['final_response']}")
        
        # Показываем статус системы
        status = pipeline.get_system_status()
        print(f"Объектов в системе: {status['total_objects']}")

if __name__ == "__main__":
    test_thinking_pipeline() 