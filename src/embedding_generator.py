import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    """
    Класс для генерации эмбеддингов предложений с использованием моделей SentenceTransformer.
    """
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        Инициализирует генератор эмбеддингов.

        Args:
            model_name (str): Название модели из библиотеки sentence-transformers.
                              По умолчанию 'paraphrase-multilingual-mpnet-base-v2' (768 измерений).
                              Другой вариант: 'all-MiniLM-L6-v2' (384 измерения, быстрее).
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Successfully loaded model '{model_name}' with embedding dimension {self.embedding_dim}.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{model_name}': {e}")
            print("Attempting to load on CPU explicitly if CUDA error occurred.")
            # Попытка загрузить на CPU, если была ошибка CUDA (например, нет CUDA драйверов)
            if self.device == 'cuda':
                try:
                    self.device = 'cpu'
                    print(f"Retrying on device: {self.device}")
                    self.model = SentenceTransformer(model_name, device=self.device)
                    self.embedding_dim = self.model.get_sentence_embedding_dimension()
                    print(f"Successfully loaded model '{model_name}' on CPU with embedding dimension {self.embedding_dim}.")
                except Exception as e_cpu:
                    print(f"Failed to load model '{model_name}' on CPU as well: {e_cpu}")
                    raise e_cpu # Перевыбрасываем ошибку, если и на CPU не удалось
            else:
                raise e # Перевыбрасываем исходную ошибку, если она не связана с CUDA


    def get_embeddings(self, sentences: list[str] | str) -> np.ndarray | None:
        """
        Генерирует эмбеддинги для одного или нескольких предложений.

        Args:
            sentences (list[str] | str): Одно предложение (str) или список предложений (list[str]).

        Returns:
            np.ndarray | None: NumPy массив с эмбеддингами.
                               Форма: (N, embedding_dim) для списка из N предложений,
                               или (embedding_dim,) для одного предложения.
                               Возвращает None, если модель не была загружена.
        """
        if not hasattr(self, 'model'):
            print("Model not loaded. Cannot generate embeddings.")
            return None

        if isinstance(sentences, str):
            sentences = [sentences]

        try:
            embeddings = self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
            return embeddings if len(sentences) > 1 else embeddings[0]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None

if __name__ == '__main__':
    print("Initializing EmbeddingGenerator...")
    # Используем модель, которая дает 768 измерений, как просил пользователь
    # 'paraphrase-multilingual-mpnet-base-v2' is a good multilingual model with 768 dim
    # 'sentence-transformers/LaBSE' is another good one, multilingual, 768 dim
    try:
        # generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2') # 384 dim
        generator = EmbeddingGenerator(model_name='paraphrase-multilingual-mpnet-base-v2') # 768 dim


        if hasattr(generator, 'model'):
            print(f"\n--- Example with a single sentence (dim: {generator.embedding_dim}) ---")
            sentence1 = "Это тестовое предложение."
            embedding1 = generator.get_embeddings(sentence1)
            if embedding1 is not None:
                print(f"Sentence: {sentence1}")
                print(f"Embedding shape: {embedding1.shape}")
                # print(f"Embedding (first 5 values): {embedding1[:5]}")

            print("\n--- Example with a list of sentences ---")
            sentences_list = [
                "Первое предложение для теста.",
                "Второе предложение немного длиннее.",
                "И третье."
            ]
            embeddings_list = generator.get_embeddings(sentences_list)
            if embeddings_list is not None:
                print(f"Sentences: {sentences_list}")
                print(f"Embeddings shape: {embeddings_list.shape}")
                # print(f"Embedding for the first sentence (first 5 values): {embeddings_list[0][:5]}")

            print("\n--- Example with Russian sentences ---")
            russian_sentences = [
                "Привет, мир!",
                "Как твои дела?",
                "Эта модель должна понимать русский язык."
            ]
            russian_embeddings = generator.get_embeddings(russian_sentences)
            if russian_embeddings is not None:
                print(f"Sentences: {russian_sentences}")
                print(f"Embeddings shape: {russian_embeddings.shape}")
                # print(f"Embedding for 'Привет, мир!' (first 5 values): {russian_embeddings[0][:5]}")

            # Проверка работы с другим языком, если модель мультиязычная
            if "multilingual" in generator.model.name_or_path or "LaBSE" in generator.model.name_or_path :
                print("\n--- Example with English sentences (multilingual model) ---")
                english_sentences = [
                    "Hello, world!",
                    "How are you doing?"
                ]
                english_embeddings = generator.get_embeddings(english_sentences)
                if english_embeddings is not None:
                    print(f"Sentences: {english_sentences}")
                    print(f"Embeddings shape: {english_embeddings.shape}")
                    # print(f"Embedding for 'Hello, world!' (first 5 values): {english_embeddings[0][:5]}")
        else:
            print("EmbeddingGenerator could not be initialized with a model.")

    except Exception as e:
        print(f"An error occurred during the example run: {e}")
        print("Please ensure you have an internet connection for the first model download,")
        print("and that the model name is correct.")
        print("If you are seeing CUDA errors, try running on CPU or ensure CUDA toolkit is correctly installed.")

    print("\nEmbeddingGenerator script finished.")
