import torch
import torch.nn as nn
import math
from typing import Tuple

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

def apply_rotary_emb_torch(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Select frequencies corresponding to the token positions
    # position_ids needs to be passed or derived from input
    # For now, assuming position_ids is available in the scope or passed
    # This part needs to be handled carefully when integrating into a larger model
    # For a general module, position_ids should be an input to forward method
    # or generated internally based on sequence length.
    # For this standalone module, we'll assume it's passed.
    
    # The original code used a global position_ids, which is not ideal for a module.
    # We will modify the forward method of SimplifiedLlama4Attention to accept position_ids.
    
    # To make this helper function truly independent, it should not rely on a global position_ids.
    # Let's assume freqs_cis is already indexed by position_ids before being passed here.
    # Or, better, pass position_ids to this function.
    
    # Re-evaluating: The original `apply_rotary_emb_torch` takes `freqs_cis` and `position_ids`
    # from the global scope. This is bad practice for a module. 
    # The `freqs_cis` should be indexed by `position_ids` *before* calling this function,
    # or `position_ids` should be an argument to this function.
    # For now, I will keep the original structure but note this for the full model integration.
    
    # Let's assume freqs_cis is already indexed by position_ids for this helper.
    # The `freqs_cis.to(xq.device)[position_ids]` line is problematic if position_ids is global.
    # It should be `freqs_cis.to(xq.device)[pos_ids_for_this_batch]`
    
    # For the purpose of refactoring, I will assume position_ids is passed to the attention module's forward
    # and then passed to this helper, or freqs_cis is pre-indexed.
    
    # Let's adjust simple_rope_calculation to return the full freqs_cis, and then index it in the attention module.
    # The apply_rotary_emb_torch will then just take the indexed freqs_cis.
    
    # Corrected approach for apply_rotary_emb_torch:
    # It should take freqs_cis (full table) and position_ids.
    # The indexing should happen inside this function or before.
    
    # Given the current structure, the simplest is to pass position_ids to this function.
    # However, the original code had `freqs_cis = freqs_cis.to(xq.device)[position_ids]`
    # which implies `position_ids` is available. I will make `position_ids` an argument.
    
    # This is the corrected signature and usage for apply_rotary_emb_torch
    # to be truly modular.
    
    # The original code had a global `position_ids` which is bad.
    # The `SimplifiedLlama4Attention` class's forward method already takes `position_ids`.
    # So, the `apply_rotary_emb_torch` function should also take `position_ids`.
    
    # Let's assume `freqs_cis` is the full precomputed table, and `position_ids` are the indices.
    
    # Select frequencies corresponding to the token positions
    freqs_cis_indexed = freqs_cis.to(xq.device)[position_ids]
    freqs_cis_indexed = freqs_cis_indexed[:, None, :, :]  # Broadcast to match shape
    
    # Convert to complex for rotation
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_broadcast = freqs_cis_indexed[..., :xq_.shape[-1]]
    # Apply rotation
    rotated_xq = xq_ * freqs_cis_broadcast
    rotated_xk = xk_ * freqs_cis_broadcast
    # Convert back to real tensors
    xq_out = torch.view_as_real(rotated_xq).flatten(3)
    xk_out = torch.view_as_real(rotated_xk).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ------------------------
# Optional Q/K normalization
# ------------------------
class SimpleL2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

# ------------------------
# Repeat K/V for multi-query or grouped-query attention
# ------------------------
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

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
        # Pass position_ids to apply_rotary_emb_torch
        q_rope, k_rope = apply_rotary_emb_torch(q, k, self.freqs_cis, position_ids)

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

if __name__ == "__main__":
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