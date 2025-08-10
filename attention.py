#import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple,Optional


#configuration
hidden_size = 128 # dimensionality of the model's hidden states
num_attention_heads = 16 # total number of query heads
num_key_value_heads = 4 # number of key/value heads (for GQA)
head_dim = hidden_size // num_attention_heads #dimension of each attention head
max_pos_embedding = 256 #maximum sequence length the model expects
rope_theta = 1000.0 # base for rotary position embedding frequency calculation
rms_norm_eps = 1e-5 #epsilon for RMSNorm
attention_bias = False #whether to use bias in Q(query)
attention_dropout = 0.0 #dropout probability for attention weights
use_qk_norm = True #whether to apply L2 normalization to Q(query) and K(key) before attention 

#example input
batch_size =2 
sequence_length = 10
hidden_states = torch.randn(batch_size,sequence_length,hidden_size)
pos_ids = torch.arange(0,sequence_length).unsqueeze(0).repeat(batch_size,1)

#simple casual mask (upper trianngular)
attention_mask = torch.triu(torch.ones(sequence_length,sequence_length)*-torch.inf,diagonal=1)
attention_mask = attention_mask.unsqueeze(0).unsqueenze(0) #shape: (1,1,sequence_length,sequence)
attention_mask = attention_mask.expand(batch_size,1,-1,-1) #shape: (1,1,sequence_length,sequence)

print("Configuration:")
print(f"  hidden_size: {hidden_size}")
print(f"  num_attention_heads: {num_attention_heads}")
print(f"  num_key_value_heads: {num_key_value_heads}")
print(f"  head_dim: {head_dim}")

print("\nSample Input Shapes:")
print(f"  hidden_states: {hidden_states.shape}")
print(f"  position_ids: {pos_ids.shape}")
print(f"  attention_mask: {attention_mask.shape}")


# Q,K,V projection
# Llama 4 uses Grouped-Query Attention (GQA). This means there are fewer K and V heads than Q heads. The `num_key_value_groups` tells us how many Q heads share the same K and V head. This reduces computation and memory requirements.

# Define projection layers
q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

# Calculate projections
query_states = q_proj(hidden_states)
key_states = k_proj(hidden_states)
value_states = v_proj(hidden_states)

# Reshape Q, K, V for multi-head attention
# Target shape: (batch_size, num_heads, sequence_length, head_dim)
query_states = query_states.view(batch_size, sequence_length, num_attention_heads, head_dim).transpose(1, 2)
key_states = key_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)
value_states = value_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)

print("Projected Shapes:")
print(f"  query_states: {query_states.shape}") # (batch_size, num_attention_heads, sequence_length, head_dim)
print(f"  key_states: {key_states.shape}")     # (batch_size, num_key_value_heads, sequence_length, head_dim)
print(f"  value_states: {value_states.shape}")   # (batch_size, num_key_value_heads, sequence_length, head_dim)

num_key_value_groups = num_attention_heads // num_key_value_heads
print(f"\nNum Key/Value Groups (Q heads per K/V head): {num_key_value_groups}")


##implementing rotary positional embedding(RoPE)

def sinple_rope_calculation(dim,max_seq_len,base=1000.0,device=None):
    """Calculation simplified RoPE frequencies"""
    inv_freq = 1.0 / (base **(torch.arange(0,dim,2,device=device).float()/ dim))
    t = torch.arange(max_seq_len,device=device).type_as(inv_freq)
    freqs = new_func(inv_freq,t)
    emb = torch.cat((freqs,freqs))
    freq_cos = emb.cos() #real part
    freq_sin = emb.sin() # imaginary part
    #combining the real and imaginary parts to from complex numbers
    freqs_cis = torch.complex(freq_cos,freq_sin)
    return freqs_cis

def new_func(inv_freq,t):
    freqs = torch.outer(t,inv_freq)
    return freqs

def apply_rotary_emb_torch(
        xq:torch.Tensor, # Query tensor,shape (batch, num_heads, seq_len, head_dim)
        xk: torch.Tensor,      # Key tensor, shape (batch, num_heads, seq_len, head_dim)
        freqs_cis: torch.Tensor, # Precomputed complex rotations, shape (max_seq_len, head_dim)
) -> Tuple[torch.Tensor,torch.Tensor]:
    """Applies RoPE rotations to Q and K using torch complex numbers."""

    #ensuring freq_cis is  one the right device (CPU/GPU)
    freqs_cis = freqs_cis.to(xq.device)

    #selecting the correct rotation vectors for current sequence positions
    freqs_cis =freqs_cis[pos_ids]

    #add a dimension for broadcasting across attention heads
    freqs_cis = freqs_cis[:,None,:,:] # Now shape: (batch, 1, seq_len, head_dim), complex


    #reshaping Q  and K to view adjacent pairs as complex numbers
    xq_ = torch.view_as_complex(xq.float(*xq.shape[:-1],-1,2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    #selecting the necessary part of freq_cis for complex math

    freqs_cis_broadcast = freqs_cis[...,:xq_.shape[-1]] #slices the last dim
    
