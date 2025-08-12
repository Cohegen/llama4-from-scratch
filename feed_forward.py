#importing libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,Optional

#defining configuration parameters relevnt to the FeedForward Network
hidden_size= 128
ffn_intermediate_ratio = 8/3
multiple_of =32
intermediate_size = int(hidden_size*ffn_intermediate_ratio)
intermediate_size = ((intermediate_size + multiple_of-1) // multiple_of)*multiple_of

hidden_act = "silu"
rms_norm_eps = 1e-5
ffn_bias = False 

#sample input (output of attention + residual)
batch_size = 2
sequence_length = 10
#state before the post-attention layernorm
input_to_ffn_block = torch.randn(batch_size,sequence_length,hidden_size)


print("Configuration:")
print(f"  hidden_size: {hidden_size}")
print(f"  intermediate_size: {intermediate_size} (Calculated from ratio {ffn_intermediate_ratio:.2f}, multiple of {multiple_of})")
print(f"  hidden_act: {hidden_act}")
print(f"  rms_norm_eps: {rms_norm_eps}")

print("\nSample Input Shape (Before FFN Block Norm):")
print(f"  input_to_ffn_block: {input_to_ffn_block.shape}")

## 2 Pre-Normalization (post-attention layernorm)

#RMSNorm Implementation
class SimplifiedRMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) #creating learnable parameters
        self.variance_epsilon = eps

    def forward(self,hidden_states):
        input_dtype= hidden_states.dtype()
        hidden_states = hidden_states.to(torch.float32)#calculating in float 32 for stability
        #calculating variance across dimensions
        variance = hidden_states.pow(2).mean(-1,keepdim=True)
        #normalizing inputs of hidden states
        hidden_states =hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        #applying the learnable weights and casting back to original dtype
        return (self.weight*hidden_states).to(input_dtype)
    
#instantiating and applying the normalization
post_attn_norm = SimplifiedRMSNorm(hidden_size,eps=rms_norm_eps)
normalized_hidden_states = post_attn_norm(input_to_ffn_block)

print("Shape after Post-Attention RMSNorm:")
print(f"normalized_hidden_states:{normalized_hidden_states.shape}")

## 3 Implementing The FeedForward Network (Multi-layer Perceptron with Gated Linear Unit)
##definig FFN layers

gate_proj = nn.Linear(hidden_size,intermediate_size,bias=ffn_bias) #projects the input into the intermediate_size
up_proj = nn.Linear(hidden_size,intermediate_size,bias=ffn_bias) #projects the input to the intermediate_size
down_proj = nn.Linear(intermediate_size,hidden_size,bias=ffn_bias)#projects the result back down to the hidden_size

#defining the activation function (SiLU/Swish)
if hidden_act == "silu":
    activation_fn = nn.SiLU()
else:
    #adding other activations if needed,otherwise raise error
    raise NotImplementedError(f"Activation {hidden_act} not implemented here")
