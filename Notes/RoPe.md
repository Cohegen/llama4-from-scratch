## Rotary Positional Embeddings

### Overview

Rotary positional embeddings (RoPE) are an innovative method for encoding positional information in transformer models. Traditional approaches—absolute and relative positional embeddings—each have limitations. Absolute positional encodings are fixed and do not generalize well to sequences longer than those seen during training, while relative encodings can capture relationships between tokens but may be less efficient or harder to implement.

RoPE combines the strengths of both approaches by encoding positional information through rotation in the embedding space. This allows the model to simultaneously grasp both the absolute position of tokens and their relative distances, leading to improved performance and generalization.

### How RoPE Works

![Output example:](../assets/rope_example.webp)

Instead of simply adding a positional vector, RoPE applies a rotation to each word's embedding vector. For example, consider a two-dimensional word vector for the token "dog" (as in the diagram above):

- The angle of rotation (θ) is proportional to the token's position:  
  - For the first position: θ  
  - For the second position: 2θ  
  - ...and so on, up to the last token.
- This rotation is applied in the multi-head attention mechanism, specifically to the query and key vectors, allowing the attention scores to be sensitive to relative positions.

#### Mathematical Intuition

RoPE uses complex numbers or 2D real-valued rotations to encode position. For a vector dimension $d$, it groups the vector into pairs and rotates each pair by a position-dependent angle. This can be interpreted as multiplying the vector by a rotation matrix or a complex exponential.

### Benefits of RoPE

1. **Relative Positional Awareness:**  
   Unlike absolute embeddings, RoPE naturally encodes relative positions between tokens. This is crucial for models like transformers, enabling them to generalize better across varying input lengths.

2. **Improved Generalization:**  
   By preserving relative distances, RoPE helps models generalize to unseen text lengths and domains more effectively than absolute positional encodings.

3. **Efficiency:**  
   RoPE does not require large embedding tables as with learned absolute embeddings. Instead, it uses simple sinusoidal functions and rotations, making it computationally light.

4. **Stability:**  
   Adding tokens at the end of a sentence does not affect the vectors for words at the beginning, which facilitates efficient caching and incremental processing.

5. **Compatibility:**  
   RoPE can be easily integrated into existing transformer architectures with minimal changes to the codebase.

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

  - $p$: token's position
  - $i$: index of the 2D pair
  - $d$: full vector dimension

- This is the same frequency scaling trick from `sinusoidal embeddings`.

#### Applying the Rotation

1. **Compute the rotation angle for each pair:**  
   For each 2D pair $(x, y)$ in the query or key vector, the rotated values are:
  
   This rotates the pair by the computed angle, encoding position directly into the representation.

2. **Apply this rotation separately to Query and Key vectors:**  
   The query and key for each token at position $p$ are rotated by their respective angles, ensuring that the positional relationship is encoded in the attention mechanism itself.

### Where RoPE Is Used

RoPE has become a standard positional encoding technique in many large language models, such as GPT-NeoX, Llama, and others. Its ability to handle long sequences and facilitate relative attention makes it especially useful in contexts where input lengths vary greatly.

### Further Reading & References

- [Rotary Position Embedding (RoPE) paper](https://arxiv.org/abs/2104.09864)
- [Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [Sinusoidal Positional Encodings](https://arxiv.org/abs/1706.03762)
- [GPT-NeoX RoPE implementation](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/positional_embeddings.py)

---

RoPE's core idea is to apply these rotations to the vector pairs, encoding positional information directly into the attention mechanism and thus improving both efficiency and contextual awareness in transformer models. It is a simple yet powerful modification that helps models generalize better and handle longer contexts.

