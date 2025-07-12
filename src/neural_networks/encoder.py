import numpy as np

class MockEncoder:
    """
    A mock implementation of the Encoder neural network.
    It does not use a real transformer model. Instead, it returns a fixed-size
    random vector for any given text, simulating the output of a sentence transformer.
    """
    def __init__(self, vector_size=768):
        """
        Initializes the mock encoder.
        Args:
            vector_size (int): The dimensionality of the vector to generate.
        """
        self.vector_size = vector_size
        print("NEURAL_NETWORKS: Initialized MockEncoder.")

    def encode(self, text: str) -> np.ndarray:
        """
        "Encodes" a text into a random vector.

        Args:
            text (str): The input text (ignored in this mock implementation).

        Returns:
            np.ndarray: A numpy array of shape (vector_size,).
        """
        print(f"ENCODER: Received text: '{text}'. Generating mock vector.")
        # In a real implementation, this would be:
        #   return model.encode(text)
        return np.random.rand(self.vector_size)
