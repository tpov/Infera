import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import Optional, Dict, Any
import numpy as np

class FinalGenerativeNetwork(nn.Module):
    """
    Финальная генеративная нейросеть (5-й этап)
    Преобразует 768-мерный вектор в естественно-языковой ответ
    Использует замороженный BART
    """
    
    def __init__(self, 
                 bart_model_name: str = "facebook/bart-base",
                 max_length: int = 128):
        super().__init__()
        
        self.max_length = max_length
        self.vector_dim = 768  # Фиксированный размер входа
        
        # Загружаем BART модель и токенизатор
        self.tokenizer = BartTokenizer.from_pretrained(bart_model_name)
        self.bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
        
        # Замораживаем веса BART
        for param in self.bart_model.parameters():
            param.requires_grad = False
        
        # Слой для преобразования 768-мерного вектора в эмбеддинги BART
        self.vector_to_embedding = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768)
        )
        
        # Слой нормализации
        self.layer_norm = nn.LayerNorm(768)
        
        # Заменяем эмбеддинг слой BART на наш кастомный
        self.custom_embedding = nn.Linear(768, self.bart_model.config.d_model)
        
    def forward(self, input_vector: torch.Tensor, 
                target_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Генерирует ответ на основе 768-мерного вектора
        
        Args:
            input_vector: 768-мерный вектор от промежуточной сети
            target_text: Целевой текст для обучения (опционально)
        
        Returns:
            Dict с сгенерированным текстом или loss
        """
        
        # Проверяем размер входного вектора
        assert input_vector.shape[-1] == 768, f"Ожидается 768-мерный вектор, получен: {input_vector.shape}"
        
        # Преобразуем вектор в эмбеддинги
        vector_embedding = self.vector_to_embedding(input_vector)
        vector_embedding = self.layer_norm(vector_embedding)
        
        # Расширяем до размера последовательности BART
        batch_size = input_vector.size(0)
        sequence_length = 1  # Начинаем с одного токена
        
        # Создаем расширенные эмбеддинги
        expanded_embeddings = vector_embedding.unsqueeze(1).expand(batch_size, sequence_length, -1)
        
        # Преобразуем в размерность BART
        bart_embeddings = self.custom_embedding(expanded_embeddings)
        
        if target_text is not None:
            # Режим обучения
            target_tokens = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Создаем кастомные attention маски
            attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)
            
            # Используем BART для вычисления loss
            outputs = self.bart_model(
                inputs_embeds=bart_embeddings,
                attention_mask=attention_mask,
                labels=target_tokens['input_ids']
            )
            
            return {"loss": outputs.loss}
        else:
            # Режим генерации
            # Создаем начальные токены для генерации
            start_token_id = self.tokenizer.bos_token_id
            input_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long)
            
            # Генерируем текст
            generated_ids = self.bart_model.generate(
                inputs_embeds=bart_embeddings,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.9
            )
            
            # Декодируем результат
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            return {"generated_text": generated_text[0] if len(generated_text) == 1 else generated_text}

class FinalGenerativeNetworkWrapper:
    """
    Обертка для удобного использования финальной генеративной сети
    """
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path:
            self.model = FinalGenerativeNetwork()
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model = FinalGenerativeNetwork()
        
        self.model.eval()
    
    def generate_response(self, input_vector: torch.Tensor) -> str:
        """
        Генерирует ответ на основе 768-мерного вектора
        
        Args:
            input_vector: 768-мерный вектор от промежуточной сети
        
        Returns:
            str: Сгенерированный ответ
        """
        with torch.no_grad():
            result = self.model(input_vector)
            return result["generated_text"]
    
    def train_step(self, input_vector: torch.Tensor, target_text: str) -> float:
        """
        Выполняет один шаг обучения
        
        Args:
            input_vector: 768-мерный входной вектор
            target_text: Целевой текст
        
        Returns:
            float: Значение loss
        """
        self.model.train()
        result = self.model(input_vector, target_text)
        return result["loss"].item()
    
    def save_model(self, path: str):
        """Сохраняет модель"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Загружает модель"""
        self.model.load_state_dict(torch.load(path))

class ResponseGenerator:
    """
    Полный генератор ответов, объединяющий промежуточную и финальную сети
    """
    
    def __init__(self, 
                 intermediate_model_path: Optional[str] = None,
                 final_model_path: Optional[str] = None):
        
        self.intermediate_network = IntermediateNetworkWrapper(intermediate_model_path)
        self.final_network = FinalGenerativeNetworkWrapper(final_model_path)
    
    def generate_response(self, user_query: str, controller_result: Dict[str, Any]) -> str:
        """
        Генерирует полный ответ на основе запроса и результата контроллера
        
        Args:
            user_query: Запрос пользователя
            controller_result: Результат контроллера
        
        Returns:
            str: Сгенерированный ответ
        """
        # 4-й этап: Промежуточная сеть → 768-мерный вектор
        intermediate_vector = self.intermediate_network.process(user_query, controller_result)
        
        # 5-й этап: Финальная генеративная сеть → ответ
        response = self.final_network.generate_response(intermediate_vector)
        
        return response

if __name__ == "__main__":
    # Тестируем финальную генеративную сеть
    from intermediate_network import IntermediateNetworkWrapper
    
    # Создаем тестовые данные
    user_query = "Создай робота с возрастом 25 и стоимостью -100"
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
    
    # Тестируем полный пайплайн
    response_generator = ResponseGenerator()
    
    try:
        response = response_generator.generate_response(user_query, controller_result)
        print(f"Запрос: {user_query}")
        print(f"Ответ: {response}")
    except Exception as e:
        print(f"Ошибка при генерации: {e}")
        print("Это нормально, так как модели не обучены") 