import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_scheduler
from torch.optim import AdamW
from encoder import Encoder
from decoder import Decoder
import numpy as np
import random
from tqdm import tqdm
import os

# --- Configuration ---
MODEL_DIR = "trained_models"
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_finetuned")
ENCODER_MODEL_NAME = 'all-MiniLM-L6-v2'
DECODER_MODEL_NAME = 't5-base'
BATCH_SIZE = 8
NUM_EPOCHS = 1  # We'll run one epoch at a time in our infinite loop
LEARNING_RATE = 5e-5

def get_wikipedia_data(num_samples=100):
    """
    Placeholder function to get text data.
    In a real scenario, this would download and parse Wikipedia articles.
    For now, it returns a list of sample sentences.
    """
    # In a real implementation, you might use the 'wikipedia' library
    # pip install wikipedia
    # import wikipedia
    # ... logic to fetch and clean articles ...

    sample_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is a branch of computer science.",
        "The capital of France is Paris.",
        "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
        "The solar system consists of the Sun and the objects that orbit it.",
        "Shakespeare was a renowned English playwright and poet.",
        "The Great Wall of China is a series of fortifications made of stone, brick, and other materials.",
        "The theory of relativity was developed by Albert Einstein.",
        "Water is composed of hydrogen and oxygen atoms.",
        "The internet has revolutionized communication and access to information."
    ]

    # Repeat and shuffle to get more data
    data = (sample_sentences * (num_samples // len(sample_sentences) + 1))[:num_samples]
    random.shuffle(data)
    return data

def main():
    """
    Main function to run the training loop.
    """
    print("--- Initializing Models ---")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Initialize encoder
    encoder = Encoder(model_name=ENCODER_MODEL_NAME)

    # Initialize or load fine-tuned decoder
    if os.path.exists(DECODER_PATH):
        print(f"Loading fine-tuned decoder from {DECODER_PATH}...")
        decoder = Decoder(model_name=DECODER_PATH)
    else:
        print(f"Initializing new decoder from {DECODER_MODEL_NAME}...")
        decoder = Decoder(model_name=DECODER_MODEL_NAME)

    if not encoder.model or not decoder.model or not decoder.tokenizer:
        print("A model failed to load. Exiting training.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.model.to(device)
    print(f"Using device: {device}")

    # --- Optimizer and Scheduler ---
    optimizer = AdamW(decoder.model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0
    training_step = 0

    print("\n--- Starting Training Loop ---")
    while True:
        print(f"\n--- Training Step {training_step + 1} ---")

        # 1. Generate Data
        print("Generating training data...")
        texts = get_wikipedia_data(num_samples=100) # Smaller batch for each step

        # Create input embeddings and target token IDs
        input_vectors = encoder.encode(texts)

        # We need to adapt the vector size
        hidden_size = decoder.model.config.d_model
        adapted_vectors = np.zeros((len(input_vectors), hidden_size))
        for i, vec in enumerate(input_vectors):
            vec_len = len(vec)
            if vec_len > hidden_size:
                adapted_vectors[i] = vec[:hidden_size]
            else:
                adapted_vectors[i, :vec_len] = vec

        # Tokenize the original texts to be the labels
        labels = decoder.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).input_ids

        # Create dataset and dataloader
        dataset = TensorDataset(torch.tensor(adapted_vectors, dtype=torch.float32), labels)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

        num_training_steps = NUM_EPOCHS * len(dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        # 2. Train for one epoch
        decoder.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            batch_inputs, batch_labels = [b.to(device) for b in batch]

            # T5 expects inputs_embeds to be (batch_size, seq_length, hidden_size)
            # We are using the vector as a single "token" embedding
            inputs_embeds = batch_inputs.unsqueeze(1)

            outputs = decoder.model(inputs_embeds=inputs_embeds, labels=batch_labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # 3. Evaluate (simplified accuracy)
        # A real evaluation would use a separate validation set and metrics like BLEU or ROUGE.
        # Here, we'll just check if the model can decode one of the training samples.
        decoder.model.eval()
        with torch.no_grad():
            sample_text = texts[0]
            sample_vector = torch.tensor(adapted_vectors[0], dtype=torch.float32).to(device)

            generated_ids = decoder.model.generate(inputs_embeds=sample_vector.unsqueeze(0).unsqueeze(0))
            decoded_text = decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Simplified "accuracy" - just a check if the output is not empty
            current_accuracy = 1.0 if decoded_text else 0.0
            print(f"Sample Original:   '{sample_text}'")
            print(f"Sample Decoded:    '{decoded_text}'")
            print(f"Current 'Accuracy' (1.0 if not empty): {current_accuracy}")


        # 4. Save Model if it's better
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            print(f"New best accuracy! Saving model to {DECODER_PATH}...")
            decoder.model.save_pretrained(DECODER_PATH)
            decoder.tokenizer.save_pretrained(DECODER_PATH)

        training_step += 1

if __name__ == "__main__":
    main()
