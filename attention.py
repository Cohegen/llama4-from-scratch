import torch
import torch.nn as nn
import math
from typing import Tuple

# ------------------------
# Configuration parameters
# ------------------------
hidden_size = 128
num_attention_heads = 16
num_key_value_heads = 4
head_dim = hidden_size // num_attention_heads
max_position_embeddings = 256
rope_theta = 10000.0
attention_bias = False
use_qk_norm = True

# ------------------------
# Dummy input tensors
# ------------------------
batch_size = 2
sequence_length = 10

# Random hidden states simulating token embeddings
hidden_states = torch.randn(batch_size, sequence_length, hidden_size)

# Position IDs for each token in the sequence
position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(batch_size, 1)

# Create causal attention mask to prevent attending to future tokens
attention_mask = torch.triu(torch.ones(sequence_length, sequence_length) * -torch.inf, diagonal=1)
attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

# ------------------------
# Projection layers for Q, K, V, and output
# ------------------------
q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

# ------------------------
# Compute Q, K, V tensors
# ------------------------
query_states = q_proj(hidden_states).view(batch_size, sequence_length, num_attention_heads, head_dim).transpose(1, 2)
key_states = k_proj(hidden_states).view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)
value_states = v_proj(hidden_states).view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)

# ------------------------
# Rotary Positional Embeddings (RoPE)
# ------------------------
def simple_rope_calculation(dim, max_seq_len, base=10000.0, device=None):
    # Compute inverse frequency for sinusoidal components
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # Time positions for each token
    t = torch.arange(max_seq_len, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    # Stack cosine and sine components
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.complex(emb.cos(), emb.sin())

def apply_rotary_emb_torch(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Select frequencies corresponding to the token positions
    freqs_cis = freqs_cis.to(xq.device)[position_ids]
    freqs_cis = freqs_cis[:, None, :, :]  # Broadcast to match shape
    # Convert to complex for rotation
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_broadcast = freqs_cis[..., :xq_.shape[-1]]
    # Apply rotation
    rotated_xq = xq_ * freqs_cis_broadcast
    rotated_xk = xk_ * freqs_cis_broadcast
    # Convert back to real tensors
    xq_out = torch.view_as_real(rotated_xq).flatten(3)
    xk_out = torch.view_as_real(rotated_xk).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# Compute RoPE embeddings and apply to Q, K
freqs_cis = simple_rope_calculation(head_dim, max_position_embeddings, base=rope_theta, device=hidden_states.device)
query_states_rope, key_states_rope = apply_rotary_emb_torch(query_states, key_states, freqs_cis)

# ------------------------
# Optional Q/K normalization
# ------------------------
class SimpleL2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

if use_qk_norm:
    qk_norm = SimpleL2Norm()
    query_states_final = qk_norm(query_states_rope)
    key_states_final = qk_norm(key_states_rope)
else:
    query_states_final = query_states_rope
    key_states_final = key_states_rope

# ------------------------
# Repeat K/V for multi-query or grouped-query attention
# ------------------------
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

num_key_value_groups = num_attention_heads // num_key_value_heads
key_states_repeated = repeat_kv(key_states_final, num_key_value_groups)
value_states_repeated = repeat_kv(value_states, num_key_value_groups)

# ------------------------
# Scaled Dot-Product Attention
# ------------------------
attn_weights = torch.matmul(query_states_final, key_states_repeated.transpose(2, 3))
attn_weights *= 1.0 / math.sqrt(head_dim)  # Scale by head dimension
if attention_mask is not None:
    causal_mask = attention_mask[:, :, :, :key_states_repeated.shape[-2]]
    attn_weights += causal_mask
attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)

# Weighted sum of values
attn_output = torch.matmul(attn_weights, value_states_repeated)

# ------------------------
# Final projection back to hidden size
# ------------------------
attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)
final_attn_output = o_proj(attn_output)

# ------------------------
# Wrap into a class for modular use
# ------------------------
class SimplifiedLlama4Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Store configuration
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.max_position_embeddings = config['max_position_embeddings']
        self.rope_theta = config['rope_theta']
        self.attention_bias = config['attention_bias']
        self.use_qk_norm = config['use_qk_norm']

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)

        # Precompute RoPE frequencies
        self.freqs_cis = simple_rope_calculation(self.head_dim, self.max_position_embeddings, base=self.rope_theta)

        # Optional normalization for Q/K
        if self.use_qk_norm:
            self.qk_norm = SimpleL2Norm()

    def forward(self, hidden_states, attention_mask, position_ids):
        bsz, seq_len, _ = hidden_states.shape

        # Q, K, V projections
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q_rope, k_rope = apply_rotary_emb_torch(q, k, self.freqs_cis.to(hidden_states.device))

        # Optional normalization
        if self.use_qk_norm:
            q_final = self.qk_norm(q_rope)
            k_final = self.qk_norm(k_rope)
        else:
            q_final, k_final = q_rope, k_rope

        # Repeat K and V for multi-head attention compatibility
        k_rep = repeat_kv(k_final, self.num_key_value_groups)
        v_rep = repeat_kv(v, self.num_key_value_groups)

        # Compute scaled dot-product attention
        attn_weights = torch.matmul(q_final, k_rep.transpose(2, 3)) * (1.0 / math.sqrt(self.head_dim))
        if attention_mask is not None:
            attn_weights += attention_mask[:, :, :, :k_rep.shape[-2]]

        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v_rep).transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)

        # Output projection
        return self.o_proj(attn_output), attn_weights

# ------------------------
# Test the class
# ------------------------
config_dict = {
    'hidden_size': hidden_size,
    'num_attention_heads': num_attention_heads,
    'num_key_value_heads': num_key_value_heads,
    'max_position_embeddings': max_position_embeddings,
    'rope_theta': rope_theta,
    'attention_bias': attention_bias,
    'use_qk_norm': use_qk_norm,
}

simplified_attn_module = SimplifiedLlama4Attention(config_dict)
final_output_simplified, final_weights_simplified = simplified_attn_module(hidden_states, attention_mask, position_ids)
print("Output shape:", final_output_simplified.shape)
print("Attention weights shape:", final_weights_simplified.shape)
