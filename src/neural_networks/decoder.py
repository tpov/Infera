import numpy as np

class MockDecoder:
    """
    A mock implementation of the Decoder network.
    It does not use a real seq2seq model. Instead, it checks for specific
    patterns in the incoming vector from the Reasoning Network and returns
    a hardcoded, human-readable string.
    """
    def __init__(self):
        print("NEURAL_NETWORKS: Initialized MockDecoder.")

    def decode(self, response_vector: np.ndarray) -> str:
        """
        "Decodes" a vector from the Reasoning Network into a text string.

        Args:
            response_vector (np.ndarray): The vector to be decoded.

        Returns:
            str: A human-readable text response.
        """
        print("DECODER: Received response vector. Decoding into text.")

        # --- Mock Logic ---
        # This logic is designed to work with the specific mock vector created
        # by the MockReasoningNetwork.

        # Check if the vector matches the "tiger attacks elephant" pattern
        # (all values are 0.5, except the first one)
        if np.all(response_vector[1:] == 0.5):
            elephant_count = int(response_vector[0])
            return (f"В стаде теперь {elephant_count} слонов и 1 тигр. "
                    "Анализ показывает, что существует высокая вероятность конфликта, "
                    "так как тигры являются хищниками для слонов.")

        # If no specific pattern is matched, return a generic response.
        else:
            return "Анализ завершен. Система рассмотрела гипотетический сценарий."
