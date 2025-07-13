import numpy as np
from sentence_transformers import SentenceTransformer

class Encoder:
    """
    A real implementation of the Encoder neural network.
    It uses a pre-trained SentenceTransformer model to generate embeddings.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the Encoder.

        Args:
            model_name (str): The name of the SentenceTransformer model to load.
        """
        self.model_name = model_name
        self.model = None
        try:
            # The model will be downloaded from Hugging Face Hub the first time.
            self.model = SentenceTransformer(self.model_name)
            print(f"ENCODER: Successfully loaded SentenceTransformer model '{self.model_name}'.")
        except Exception as e:
            print(f"ENCODER: Failed to load model '{self.model_name}'.")
            print(f"ENCODER: Error: {e}")
            print("ENCODER: The system will not be able to generate real embeddings.")


    def encode(self, text: str) -> np.ndarray:
        """
        Encodes a text into a dense vector using the loaded model.

        Args:
            text (str): The input text.

        Returns:
            np.ndarray: A numpy array representing the sentence embedding.
                        Returns a zero vector if the model is not loaded.
        """
        if self.model:
            print(f"ENCODER: Encoding text: '{text}'")
            # The encode method returns a numpy array.
            return self.model.encode(text)
        else:
            print("ENCODER: Model not loaded, returning zero vector.")
            # The size needs to be known. Most 'all-MiniLM-L6-v2' models have 384 dimensions.
            # A more robust solution would fetch this from model config.
            return np.zeros(384)
