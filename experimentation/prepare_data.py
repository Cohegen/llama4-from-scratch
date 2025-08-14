import os
import numpy as np
from bpe_tokenizer import BpeTokenizer

# Load the dataset
with open("divine_comedy.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Train the tokenizer
num_merges = 5000

# Check if a tokenizer is already saved
if not os.path.exists(f"tokenizer_vocab.json"):
    print(f"Training tokenizer with {num_merges} merges...")
    tokenizer = BpeTokenizer(corpus=[text], num_merges=num_merges)
    tokenizer.save_tokenizer("tokenizer")
else:
    print("Loading existing tokenizer...")
    tokenizer = BpeTokenizer()
    tokenizer.load_tokenizer("tokenizer")

# Encode the dataset
print("Encoding dataset...")
encoded_data = tokenizer.encode(text)

# Save to a binary file
encoded_data = np.array(encoded_data, dtype=np.uint16)
encoded_data.tofile("train.bin")

print("Dataset prepared and saved to train.bin")
print(f"Vocabulary size: {len(tokenizer.vocab)}")
print(f"Number of tokens in train.bin: {len(encoded_data)}")
