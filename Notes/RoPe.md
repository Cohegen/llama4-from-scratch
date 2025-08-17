## Rotary Positional Embeddings

### Overview

Rotary positional embeddings (RoPE) are an innovative method for encoding positional information in transformer models. Traditional approaches—absolute and relative positional embeddings—each have their limitations. Absolute embeddings assign a unique vector to each position, which does not scale well and fails to capture relative relationships between tokens. Relative embeddings focus on token distances, enhancing relational understanding but often complicating model architecture.

RoPE combines the strengths of both approaches by encoding positional information through rotation in the embedding space. This allows the model to simultaneously grasp both the absolute position of tokens and their relative relationships.

### How RoPE Works

![Output example:](../assets/rope_example.webp)

Instead of simply adding a positional vector, RoPE applies a rotation to each word's embedding vector. For example, consider a two-dimensional word vector for the token "dog" (as in the diagram above). To encode its position, RoPE rotates this vector by an angle proportional to the token's position in the sentence:

- The angle of rotation (θ) is proportional to the token's position:  
  - For the first position: θ  
  - For the second position: 2θ  
  - ...and so on, up to the last token.

### Benefits of RoPE

1. **Relative Positional Awareness:**  
   Unlike absolute embeddings, RoPE naturally encodes relative positions between tokens. This is crucial for models like transformers, enabling them to generalize better across varying input lengths.

2. **Improved Generalization:**  
   By preserving relative distances, RoPE helps models generalize to unseen text lengths and domains more effectively than absolute positional encodings.

3. **Efficiency:**  
   RoPE does not require large embedding tables as with learned absolute embeddings. Instead, it uses simple sinusoidal functions and rotations, making it computationally light.

4. **Stability:**  
   Adding tokens at the end of a sentence does not affect the vectors for words at the beginning, which facilitates efficient caching and incremental processing.

### RoPE Formulation

![Output example:](../assets/RoPE-Theta-Rotation-Formula.png)

Consider the sentence: "The pig chased the dog"

| Token   | Position $p$ |
| ------- | ------------ |
| The     | 0            |
| pig     | 1            |
| chased  | 2            |
| the     | 3            |
| dog     | 4            |

- Each token has a position $p$.
- Tokens are mapped into their respective query and key vectors of dimension $d$, grouped into 2D pairs:
    - Example for $d = 8$: $(q_0, q_1), (q_2, q_3), (q_4, q_5), (q_6, q_7)$
- The rotation angle for the $i$-th pair is calculated as:

  $$
  \theta_{p,i} = \frac{p}{10000^{2i/d}}
  $$
-Here:
  - $p --> token's position
  - $i --> which 2D pair we're rotating
  - $d --> full vector dimension
- This is the same frequency scaling trick from `sinusoidal embeddings`.

RoPE's core idea is to apply these rotations to the vector pairs, encoding positional information directly into the attention mechanism and thus improving both efficiency and contextual awareness in transformer architectures.


