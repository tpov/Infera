import torch
from typing import Dict, Any, Optional
from vectorizer import Vectorizer
from response_generator import ResponseGenerator
from command_controller import CommandController

class IntegratedPipeline:
    """
    Интегрированный пайплайн, объединяющий все 5 этапов:
    1. Векторизация входного текста
    2. Генерация команд из вектора
    3. Выполнение команд контроллером
    4. Промежуточная сеть (объединение запроса и результата)
    5. Финальная генеративная сеть (вектор → ответ)
    """
    
    def __init__(self,
                 vectorizer_model_path: Optional[str] = None,
                 command_generator_path: Optional[str] = None,
                 intermediate_model_path: Optional[str] = None,
                 final_model_path: Optional[str] = None):
        
        # Инициализируем компоненты
        self.vectorizer = Vectorizer(vectorizer_model_path)
        self.command_generator = ResponseGenerator(command_generator_path)  # Временно используем ResponseGenerator
        self.controller = CommandController()
        self.intermediate_network = ResponseGenerator(intermediate_model_path)
        self.final_network = ResponseGenerator(final_model_path)
        
        # Полный генератор ответов
        self.response_generator = ResponseGenerator(intermediate_model_path, final_model_path)
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Обрабатывает запрос пользователя через все 5 этапов
        
        Args:
            user_query: Запрос пользователя
        
        Returns:
            Dict с результатами всех этапов и финальным ответом
        """
        results = {
            "user_query": user_query,
            "stage1_vector": None,
            "stage2_commands": None,
            "stage3_controller_result": None,
            "stage4_intermediate_vector": None,
            "stage5_final_response": None,
            "success": False,
            "error": None
        }
        
        try:
            # Этап 1: Векторизация
            print("Этап 1: Векторизация...")
            vector = self.vectorizer.vectorize(user_query)
            results["stage1_vector"] = vector
            print(f"Вектор создан, размер: {vector.shape}")
            
            # Этап 2: Генерация команд
            print("Этап 2: Генерация команд...")
            # Здесь должна быть нейросеть для генерации команд
            # Пока используем простую логику
            commands = self._generate_commands_from_vector(vector)
            results["stage2_commands"] = commands
            print(f"Сгенерированы команды: {commands}")
            
            # Этап 3: Выполнение команд контроллером
            print("Этап 3: Выполнение команд...")
            controller_result = self.controller.execute_commands(commands)
            results["stage3_controller_result"] = {
                "success": controller_result.success,
                "message": controller_result.message,
                "created_objects": len(controller_result.created_objects),
                "modified_objects": len(controller_result.modified_objects),
                "computed_values": controller_result.computed_values,
                "contradictions": controller_result.contradictions,
                "warnings": controller_result.warnings
            }
            print(f"Контроллер выполнен: {controller_result.success}")
            
            # Этап 4: Промежуточная сеть
            print("Этап 4: Промежуточная сеть...")
            try:
                intermediate_vector = self.intermediate_network.process(user_query, results["stage3_controller_result"])
                results["stage4_intermediate_vector"] = intermediate_vector
                print(f"Промежуточный вектор создан: {intermediate_vector.shape}")
            except Exception as e:
                print(f"Ошибка в промежуточной сети: {e}")
                # Создаем фиктивный вектор для продолжения
                results["stage4_intermediate_vector"] = torch.randn(1, 768)
            
            # Этап 5: Финальная генеративная сеть
            print("Этап 5: Генерация ответа...")
            try:
                final_response = self.final_network.generate_response(results["stage4_intermediate_vector"])
                results["stage5_final_response"] = final_response
                print(f"Сгенерирован ответ: {final_response}")
            except Exception as e:
                print(f"Ошибка в финальной сети: {e}")
                results["stage5_final_response"] = "Извините, произошла ошибка при генерации ответа."
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = str(e)
            print(f"Ошибка в пайплайне: {e}")
        
        return results
    
    def _generate_commands_from_vector(self, vector: torch.Tensor) -> str:
        """
        Простая логика генерации команд из вектора
        В реальной системе здесь должна быть нейросеть
        """
        # Простая эвристика для демонстрации
        vector_norm = torch.norm(vector).item()
        
        if vector_norm > 0.5:
            return "create robot age 25 cost 100"
        else:
            return "create sensor name test_sensor"
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Возвращает статус системы
        """
        return {
            "total_objects": len(self.controller.system_state.objects),
            "object_types": self.controller.get_system_tree()["object_types"],
            "execution_history": len(self.controller.system_state.execution_history)
        }

def test_integrated_pipeline():
    """
    Тестирует интегрированный пайплайн
    """
    print("Тестирование интегрированного пайплайна...")
    
    # Создаем пайплайн
    pipeline = IntegratedPipeline()
    
    # Тестовые запросы
    test_queries = [
        "Создай робота с возрастом 25 и стоимостью 100",
        "Создай сенсор с именем temperature_sensor",
        "Создай устройство с количеством 5"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Тест {i}: {query}")
        print(f"{'='*50}")
        
        result = pipeline.process_query(query)
        
        print(f"Успех: {result['success']}")
        if result['error']:
            print(f"Ошибка: {result['error']}")
        
        if result['stage5_final_response']:
            print(f"Финальный ответ: {result['stage5_final_response']}")
        
        # Показываем статус системы
        status = pipeline.get_system_status()
        print(f"Объектов в системе: {status['total_objects']}")

if __name__ == "__main__":
    test_integrated_pipeline() 