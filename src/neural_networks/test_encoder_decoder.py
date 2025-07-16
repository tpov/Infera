import numpy as np
from encoder import Encoder
from decoder import Decoder

def main():
    """
    Main function to test the Encoder-Decoder pipeline.
    """
    # --- Initialization ---
    print("Initializing Encoder and Decoder...")
    encoder = Encoder()
    decoder = Decoder()
    print("Initialization complete.")

    # Check if models are loaded
    if not encoder.model or not decoder.model:
        print("A model failed to load. Exiting test.")
        return

    # --- Test Sentence ---
    input_text = "Here is a sentence that we want to encode and then decode."
    print(f"\nOriginal text: '{input_text}'")

    # --- Encoding ---
    print("\n--- Starting Encoding ---")
    encoded_vector = encoder.encode(input_text)
    print(f"Encoding complete. Vector shape: {encoded_vector.shape}")
    # Optional: print the vector to inspect it
    # print("Encoded vector:", encoded_vector)

    # --- Decoding ---
    print("\n--- Starting Decoding ---")
    decoded_text = decoder.decode(encoded_vector)
    print("Decoding complete.")

    # --- Results ---
    print("\n--- Test Results ---")
    print(f"Original text:  '{input_text}'")
    print(f"Decoded text:   '{decoded_text}'")
    print("--------------------")

    # --- Verification ---
    # A simple check to see if the output is not empty or an error message.
    if decoded_text and "[Decoder" not in decoded_text and "[Placeholder" not in decoded_text:
        print("\nTest PASSED: The decoder produced a non-empty, valid response.")
    else:
        print("\nTest FAILED: The decoder did not produce a valid response.")

if __name__ == "__main__":
    main()
