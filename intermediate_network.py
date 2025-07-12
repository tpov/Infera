import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Any, Optional
import json

class IntermediateNetwork(nn.Module):
    """
    Промежуточная нейросеть (4-й этап)
    Объединяет исходный запрос пользователя с результатом работы контроллера
    и формирует 768-мерный вектор для финальной генеративной сети
    """
    
    def __init__(self, 
                 base_model_name: str = "facebook/bart-base",
                 max_length: int = 512):
        super().__init__()
        
        self.max_length = max_length
        self.output_dim = 768  # Фиксированный размер выхода
        
        # Загружаем токенизатор и модель BART
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        
        # НЕ замораживаем веса промежуточной сети - она должна обучаться
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        # Слой для объединения запроса и результата контроллера
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 * 2, 1024),  # 768 от запроса + 768 от результата
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 768)  # Выход 768-мерный
        )
        
        # Слой нормализации
        self.layer_norm = nn.LayerNorm(768)
        
    def forward(self, user_query: str, controller_result: Dict[str, Any]) -> torch.Tensor:
        """
        Объединяет запрос пользователя с результатом контроллера
        
        Args:
            user_query: Исходный запрос пользователя
            controller_result: Результат работы контроллера (созданные объекты, ошибки и т.д.)
        
        Returns:
            torch.Tensor: 768-мерный вектор для финальной генеративной сети
        """
        
        # Подготавливаем текст результата контроллера
        controller_text = self._prepare_controller_result(controller_result)
        
        # Токенизируем запрос пользователя
        query_tokens = self.tokenizer(
            user_query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Токенизируем результат контроллера
        controller_tokens = self.tokenizer(
            controller_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Получаем эмбеддинги
        with torch.no_grad():
            query_output = self.encoder(**query_tokens)
            controller_output = self.encoder(**controller_tokens)
        
        # Берем [CLS] токены (или среднее по последовательности)
        query_embedding = query_output.last_hidden_state[:, 0, :]  # [1, 768]
        controller_embedding = controller_output.last_hidden_state[:, 0, :]  # [1, 768]
        
        # Объединяем эмбеддинги
        combined = torch.cat([query_embedding, controller_embedding], dim=1)  # [1, 1536]
        
        # Пропускаем через fusion layer
        output = self.fusion_layer(combined)
        
        # Нормализуем
        output = self.layer_norm(output)
        
        return output  # Размер: [1, 768]
    
    def _prepare_controller_result(self, controller_result: Dict[str, Any]) -> str:
        """
        Подготавливает результат контроллера в текстовый формат
        """
        text_parts = []
        
        # Добавляем информацию о созданных объектах с ошибками
        if controller_result.get('created_objects'):
            text_parts.append("Объекты с ошибками:")
            for obj in controller_result['created_objects']:
                if hasattr(obj, 'properties') and 'problem' in obj.properties:
                    problem = obj.properties['problem'].value
                    text_parts.append(f"- {obj.name}: {problem}")
        
        # Добавляем информацию о вычисленных значениях
        if controller_result.get('computed_values'):
            text_parts.append("Вычисленные значения:")
            for key, value in controller_result['computed_values'].items():
                text_parts.append(f"- {key}: {value}")
        
        # Добавляем информацию о противоречиях
        if controller_result.get('contradictions'):
            text_parts.append("Противоречия:")
            for contradiction in controller_result['contradictions']:
                text_parts.append(f"- {contradiction}")
        
        # Добавляем информацию о предупреждениях
        if controller_result.get('warnings'):
            text_parts.append("Предупреждения:")
            for warning in controller_result['warnings']:
                text_parts.append(f"- {warning}")
        
        return " ".join(text_parts) if text_parts else "Нет результатов от контроллера"

class IntermediateNetworkWrapper:
    """
    Обертка для удобного использования промежуточной сети
    """
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path:
            self.model = IntermediateNetwork()
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model = IntermediateNetwork()
        
        self.model.eval()
    
    def process(self, user_query: str, controller_result: Dict[str, Any]) -> torch.Tensor:
        """
        Обрабатывает запрос и результат контроллера
        
        Args:
            user_query: Запрос пользователя
            controller_result: Результат контроллера
        
        Returns:
            torch.Tensor: 768-мерный вектор для финальной генеративной сети
        """
        with torch.no_grad():
            return self.model(user_query, controller_result)
    
    def save_model(self, path: str):
        """Сохраняет модель"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Загружает модель"""
        self.model.load_state_dict(torch.load(path))

if __name__ == "__main__":
    # Тестируем промежуточную сеть
    network = IntermediateNetworkWrapper()
    
    # Тестовые данные
    user_query = "Создай робота с возрастом 25 и стоимостью 100"
    controller_result = {
        "created_objects": [
            type('obj', (), {
                'name': '$question$1',
                'properties': {'problem': type('prop', (), {'value': 'Отрицательное значение cost: -100'})}
            })()
        ],
        "computed_values": {"robot.age": 25},
        "contradictions": [],
        "warnings": []
    }
    
    # Получаем 768-мерный вектор
    output_vector = network.process(user_query, controller_result)
    print(f"Размер выходного вектора: {output_vector.shape}")  # Должно быть [1, 768]
    print(f"Вектор: {output_vector}") 