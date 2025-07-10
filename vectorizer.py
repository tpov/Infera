import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Инициализация модели. Модель будет загружена при первом вызове get_vector или другого метода, использующего ее.
# Это сделано для того, чтобы избежать загрузки модели при простом импорте модуля, если она не нужна сразу.
model = None
MODEL_NAME = 'all-mpnet-base-v2'

def _initialize_model():
    """Инициализирует модель SentenceTransformer, если она еще не инициализирована."""
    global model
    if model is None:
        print(f"Initializing sentence transformer model: {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        print("Model initialized.")

def get_vector(text: str) -> list[float]:
    """
    Генерирует 768-мерный вектор для заданного текста.

    Args:
        text: Входной текст.

    Returns:
        Список float, представляющий вектор.
    """
    _initialize_model()
    if not text or not isinstance(text, str):
        # Возвращаем нулевой вектор для пустого или некорректного ввода
        return [0.0] * (model.get_sentence_embedding_dimension() if model else 768)

    embedding = model.encode(text)
    return embedding.tolist()

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Рассчитывает косинусное сходство между двумя векторами.

    Args:
        vec1: Первый вектор (список float).
        vec2: Второй вектор (список float).

    Returns:
        Значение косинусного сходства (от -1 до 1).
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0  # Или можно вызвать ошибку, если векторы несовместимы

    # Преобразуем списки в numpy массивы для sklearn_cosine_similarity
    # sklearn_cosine_similarity ожидает 2D массивы
    vec1_np = np.array(vec1).reshape(1, -1)
    vec2_np = np.array(vec2).reshape(1, -1)

    similarity = sklearn_cosine_similarity(vec1_np, vec2_np)[0][0]
    return float(similarity)

if __name__ == '__main__':
    # Пример использования и тест
    print("Starting vectorizer.py self-test...")

    # Тест 1: Получение вектора
    text1 = "This is a test sentence."
    vector1 = get_vector(text1)
    print(f"Vector for '{text1}': {vector1[:5]}... (length: {len(vector1)})")
    assert len(vector1) == 768, f"Vector length is not 768, got {len(vector1)}"

    # Тест 2: Получение другого вектора
    text2 = "Another test sentence, quite different."
    vector2 = get_vector(text2)
    print(f"Vector for '{text2}': {vector2[:5]}... (length: {len(vector2)})")
    assert len(vector2) == 768, f"Vector length is not 768, got {len(vector2)}"

    # Тест 3: Сходство векторов
    text_similar1 = "The cat sat on the mat."
    text_similar2 = "A feline was resting on the rug."
    vector_sim1 = get_vector(text_similar1)
    vector_sim2 = get_vector(text_similar2)
    similarity_same_meaning = cosine_similarity(vector_sim1, vector_sim2)
    print(f"Similarity between '{text_similar1}' and '{text_similar2}': {similarity_same_meaning:.4f}")
    assert similarity_same_meaning > 0.6, "Similarity for similar sentences is too low" # Ожидаем высокое сходство

    # Тест 4: Сходство различных векторов
    similarity_different = cosine_similarity(vector1, vector2)
    print(f"Similarity between '{text1}' and '{text2}': {similarity_different:.4f}")
    assert similarity_different < 0.7, "Similarity for different sentences is too high" # Ожидаем более низкое сходство

    # Тест 5: Сходство вектора с самим собой
    similarity_self = cosine_similarity(vector1, vector1)
    print(f"Similarity of '{text1}' with itself: {similarity_self:.4f}")
    # Сходство с самим собой должно быть очень близко к 1.0
    assert 0.99 < similarity_self <= 1.0001, "Self-similarity is not close to 1.0"

    # Тест 6: Обработка пустого текста
    empty_vector = get_vector("")
    print(f"Vector for empty string: {empty_vector[:5]}... (length: {len(empty_vector)})")
    assert len(empty_vector) == 768 and all(v == 0.0 for v in empty_vector), "Empty string vector is not all zeros"

    null_vector = get_vector(None) # type: ignore
    print(f"Vector for None: {null_vector[:5]}... (length: {len(null_vector)})")
    assert len(null_vector) == 768 and all(v == 0.0 for v in null_vector), "None input vector is not all zeros"


    print("Vectorizer.py self-test completed successfully!")
