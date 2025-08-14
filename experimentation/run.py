import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F

from bpe_tokenizer import BpeTokenizer
from config import ModelConfig
from model import LlamaModel

# --- Configuration ---
BATCH_SIZE = 16
SEQUENCE_LENGTH = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20

def get_data_loader(file_path, tokenizer, batch_size, seq_length):
    # Load encoded data
    data = np.fromfile(file_path, dtype=np.uint16)
    
    inputs, targets = [], []
    for i in range(0, len(data) - seq_length, seq_length):
        inputs.append(data[i:i+seq_length])
        targets.append(data[i+1:i+seq_length+1])
        
    input_tensor = torch.tensor(inputs, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    
    dataset = TensorDataset(input_tensor, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.6, top_k=40, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    generated_ids = input_ids.tolist()[0]
    
    with torch.no_grad():
        for _ in range(max_length):
            seq_len = input_ids.shape[1]
            attention_mask = torch.triu(torch.ones(seq_len, seq_len) * -torch.inf, diagonal=1).to(device)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            
            logits = model(input_ids, attention_mask=attention_mask)[0, -1, :]
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                logits[logits < top_k_values[-1]] = -float("Inf")
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float("Inf")
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_ids.append(next_token)
            input_ids = torch.tensor([generated_ids], dtype=torch.long).to(device)
    
    return tokenizer.decode(generated_ids[len(tokenizer.encode(prompt)):])

def main():
    # 1. Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BpeTokenizer()
    tokenizer.load_tokenizer("tokenizer")
    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")

    # 2. Load data
    print("\nLoading and preparing data...")
    train_loader = get_data_loader("train.bin", tokenizer, BATCH_SIZE, SEQUENCE_LENGTH)
    print("Data loaded.")

    # 3. Load model configuration
    config = ModelConfig()
    config.vocab_size = len(tokenizer.vocab)
    print("\nModel Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # 4. Instantiate the Llama model
    print("\nInstantiating LlamaModel...")
    model = LlamaModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model successfully instantiated and moved to {device}.")

    # 5. Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 6. Training loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            attention_mask = torch.triu(torch.ones(SEQUENCE_LENGTH, SEQUENCE_LENGTH) * -torch.inf, diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(inputs.shape[0], 1, -1, -1).to(device)

            logits = model(inputs, attention_mask=attention_mask)
            loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} finished. Average Loss: {total_loss / len(train_loader):.4f}")

    print("\nTraining complete.")

    # 7. Save the trained model
    torch.save(model.state_dict(), "trained_llama_model.pth")
    print("Model saved to trained_llama_model.pth")

    # 8. Generate text with improved sampling
    print("\n--- Generating Text with Trained Model ---")
    prompt = "In the middle of the journey of our life"
    print(f"Prompt: \"{prompt}\"")
    generated_text = generate_text(model, tokenizer, prompt, max_length=200, temperature=0.6, top_k=40, top_p=0.9)
    print(f"\nGenerated Text:\n{prompt} {generated_text}")

if __name__ == "__main__":
    main()

