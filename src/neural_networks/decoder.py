import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer

class Decoder:
    """
    A real implementation of the Decoder network.
    It uses a pre-trained sequence-to-sequence model like T5.
    """
    def __init__(self, model_name='t5-base'):
        """
        Initializes the Decoder.

        Args:
            model_name (str): The name of the T5 model to load.
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            print(f"DECODER: Successfully loaded T5 model '{self.model_name}'.")
        except Exception as e:
            print(f"DECODER: Failed to load model '{self.model_name}'.")
            print(f"DECODER: Error: {e}")
            print("DECODER: The system will not be able to generate text responses.")

    def decode(self, response_vector: np.ndarray) -> str:
        """
        Decodes a vector into a text string by using it as an embedding
        to the T5 model.

        Args:
            response_vector (np.ndarray): The vector to be decoded.

        Returns:
            str: A human-readable text response.
        """
        if self.model and self.tokenizer:
            try:
                # Reshape the vector to (1, 1, embedding_dim) to represent
                # a batch of 1, with a sequence length of 1.
                # T5 expects input embeddings to be of shape (batch_size, seq_length, hidden_size)
                # The model's hidden size must match the vector's dimension.
                # For 't5-base', the hidden size is 768.
                # For 't5-small', it's 512.
                # The encoder produces 384, so we need to adjust something.
                # Let's check the actual hidden size required by the model.
                hidden_size = self.model.config.d_model

                # The encoder output is 384, but T5-base needs 768.
                # This is a mismatch. For now, let's adapt by padding or truncating.
                # This is a temporary solution. A better approach is to have a Dense layer
                # to project the encoder output to the decoder's expected input dimension.

                # Let's create a new vector of the correct size and copy the data.
                adapted_vector = np.zeros(hidden_size)
                # Copy the smaller vector's data into the new vector
                vector_len = len(response_vector)
                if vector_len > hidden_size:
                    adapted_vector = response_vector[:hidden_size]
                else:
                    adapted_vector[:vector_len] = response_vector

                inputs_embeds = torch.tensor(adapted_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                # Generate output IDs
                print("DECODER: Generating text from the adapted response vector.")
                outputs = self.model.generate(inputs_embeds=inputs_embeds, max_length=50)

                # Decode the generated IDs to a string
                decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"DECODER: Decoded text: '{decoded_text}'")
                return decoded_text
            except Exception as e:
                print(f"DECODER: An error occurred during decoding: {e}")
                return "[Decoder error]"
        else:
            return "[Decoder model not loaded]"
