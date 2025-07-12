import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import re

# Конфигурация модели
class ModelConfig:
    # Доступные предобученные модели
    MODELS = {
        'ru': 'DeepPavlov/rubert-base-cased',    # Русская модель
        'en': 'bert-base-uncased',               # Английская модель
        'bart': 'facebook/bart-base',            # BART-base для совместимости (768)
        'multilingual': 'bert-base-multilingual-cased'  # Мультиязычная модель
    }
    
    def __init__(self, model_type='bart', output_dim=None):
        """
        Инициализация конфигурации модели.
        Args:
            model_type: 'bart' (по умолчанию), 'ru', 'en', или 'multilingual'
            output_dim: Желаемая размерность выходного вектора. 
                       None = использовать оригинальную размерность модели (768)
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Неподдерживаемый тип модели. Доступные типы: {list(self.MODELS.keys())}")
        
        self.model_name = self.MODELS[model_type]
        self.output_dim = output_dim
        self.model_type = model_type

# Инициализация модели и токенизатора
model = None
tokenizer = None
config = None

def _initialize_model(model_type='bart', output_dim=None):
    """Инициализирует модель и токенизатор с заданными параметрами."""
    global model, tokenizer, config
    
    # Создаем новую конфигурацию только если параметры изменились
    if (config is None or 
        config.model_type != model_type or 
        config.output_dim != output_dim):
        
        config = ModelConfig(model_type, output_dim)
        print(f"Initializing model and tokenizer: {config.model_name}...")
        
        # Определяем устройство для вычислений
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Загружаем базовую модель и токенизатор
        base_model = AutoModel.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # ЗАМОРАЖИВАЕМ первую сеть (векторизацию) для стабильности
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Перемещаем модель на выбранное устройство
        base_model = base_model.to(device)
        
        # Определяем, нужна ли проекция в другую размерность
        base_dim = base_model.config.hidden_size  # 768 для base моделей
        need_projection = config.output_dim is not None and config.output_dim != base_dim
        
        if need_projection:
            # Создаем модель с проекционным слоем
            class ModelWithProjection(torch.nn.Module):
                def __init__(self, base_model, output_dim):
                    super().__init__()
                    self.base_model = base_model
                    self.projection = torch.nn.Linear(
                        in_features=base_dim,
                        out_features=output_dim
                    ).to(device)
                
                def forward(self, input_ids, attention_mask):
                    # Получаем выход базовой модели
                    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Для BART используем pooled_output если есть, иначе берем [CLS]
                    if hasattr(outputs, 'pooler_output'):
                        pooled = outputs.pooler_output
                    else:
                        pooled = outputs.last_hidden_state[:, 0, :]
                    
                    # Проецируем в нужную размерность
                    projected = self.projection(pooled)
                    return projected

            model = ModelWithProjection(base_model, config.output_dim)
        else:
            model = base_model
            
        model.eval()  # Переключаем в режим оценки
        print(f"Model initialized. Base dimensions: {base_dim}, " +
              f"Output dimensions: {config.output_dim if need_projection else base_dim}")

def get_vector(text: str, model_type='bart', output_dim=None) -> list[float]:
    """
    Генерирует вектор для заданного текста.
    Args:
        text: Входной текст (предложение)
        model_type: Тип модели ('bart' по умолчанию)
        output_dim: Желаемая размерность выходного вектора (None = 768)
    Returns:
        Список float значений (вектор)
    """
    _initialize_model(model_type, output_dim)
    
    if not text or not isinstance(text, str):
        out_dim = output_dim if output_dim is not None else model.config.hidden_size
        return [0.0] * out_dim

    # Очистка текста от лишних пробелов
    text = re.sub(r'\s+', ' ', text.strip())
    if not text:
        out_dim = output_dim if output_dim is not None else model.config.hidden_size
        return [0.0] * out_dim

    # Определяем устройство
    device = next(model.parameters()).device

    # Токенизация текста
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Перемещаем тензоры на нужное устройство
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Получение эмбеддинга
    with torch.no_grad():
        if isinstance(model, torch.nn.Module) and hasattr(model, 'projection'):
            # Модель с проекцией
            embedding = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            # Базовая модель
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Для BART используем pooled_output если есть
            if hasattr(outputs, 'pooler_output'):
                embedding = outputs.pooler_output
            else:
                embedding = outputs.last_hidden_state[:, 0, :]

    # Преобразование в numpy и затем в список
    embedding_np = embedding.cpu().numpy()[0]
    return embedding_np.tolist()

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Рассчитывает косинусное сходство между двумя векторами.
    Args:
        vec1: Первый вектор (список float)
        vec2: Второй вектор (список float)
    Returns:
        Значение косинусного сходства (от -1 до 1)
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    vec1_np = np.array(vec1).reshape(1, -1)
    vec2_np = np.array(vec2).reshape(1, -1)

    similarity = sklearn_cosine_similarity(vec1_np, vec2_np)[0][0]
    return float(similarity)

class Vectorizer:
    """
    Класс-обертка для векторного представления текста
    """
    
    def __init__(self, model_path: str = None):
        """
        Инициализация векторного представления
        Args:
            model_path: Путь к модели (не используется, оставлен для совместимости)
        """
        self.model_type = 'bart'  # Изменено с 'deberta' на 'bart'
        self.output_dim = 768
    
    def vectorize(self, text: str) -> torch.Tensor:
        """
        Векторизует текст
        Args:
            text: Входной текст
        Returns:
            torch.Tensor: Векторное представление
        """
        vector_list = get_vector(text, self.model_type, self.output_dim)
        return torch.tensor(vector_list, dtype=torch.float32).unsqueeze(0)  # Добавляем размерность батча

if __name__ == '__main__':
    # Тестирование на предложениях средней сложности
    print("\nТестирование BART-base на предложениях:")
    
    # Тест 1: Простые условия
    text1 = "Если цена товара больше 1000 рублей, применить скидку 10%"
    text2 = "При стоимости выше 1000 рублей дать скидку 10%"
    vec1 = get_vector(text1)  # Используем BART-base по умолчанию
    vec2 = get_vector(text2)
    sim = cosine_similarity(vec1, vec2)
    print(f"\nСходство простых условий: {sim:.4f}")
    
    # Тест 2: Составные условия
    text3 = "Отправить уведомление когда заказ готов и курьер в пути"
    text4 = "Уведомить клиента при готовности заказа и начале доставки"
    vec3 = get_vector(text3)
    vec4 = get_vector(text4)
    sim2 = cosine_similarity(vec3, vec4)
    print(f"Сходство составных условий: {sim2:.4f}")
    
    # Тест 3: Разные по смыслу предложения
    text5 = "Пользователь может войти при правильном пароле"
    text6 = "Система должна заблокировать доступ при неверном пароле"
    vec5 = get_vector(text5)
    vec6 = get_vector(text6)
    sim3 = cosine_similarity(vec5, vec6)
    print(f"Сходство разных по смыслу предложений: {sim3:.4f}")
    
    print("\nРазмерность векторов:", len(vec1))  # Должно быть 768
    print("\nВсе тесты успешно завершены!")
