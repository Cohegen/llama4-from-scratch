## Rotary Positional Embeddings

## Overview
Rotary positional embedding represents a new approach in encodingg positional information.
Traditional methods like absolute or relative, come with their limitations.
Absoluute assign a unique vector to each position, which doesn't scale well and fails to capture relative positions effectively.
Relative embeddings, focus on the distance between tokens, enhancing the model's understanding of token relationships but complcating the model architecture.
Rotary Positional Embeddings (RoPe) combines the strengths of both, that is, it encodes positional information in a way that allows the model to understand both the absolute postion of tokens and their relative distances.
This is achieved through a rotation in the embedding space.

# How RoPe works
![Output example:](../assets/rope_example.webp)
