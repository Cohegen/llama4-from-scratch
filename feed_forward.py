#importing libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Defining configuration parameters relevant to the FeedForward Network
hidden_size = 128
ffn_intermediate_ratio = 8 / 3
multiple_of = 32

# Calculate intermediate size (rounded to multiple_of)
intermediate_size = int(hidden_size * ffn_intermediate_ratio)
intermediate_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of

hidden_act = "silu"
rms_norm_eps = 1e-5
ffn_bias = False

# Sample input (output of attention + residual)
batch_size = 2
sequence_length = 10
# State before the post-attention layernorm
input_to_ffn_block = torch.randn(batch_size, sequence_length, hidden_size)

print("Configuration:")
print(f"  hidden_size: {hidden_size}")
print(f"  intermediate_size: {intermediate_size} (Calculated from ratio {ffn_intermediate_ratio:.2f}, multiple of {multiple_of})")
print(f"  hidden_act: {hidden_act}")
print(f"  rms_norm_eps: {rms_norm_eps}")

print("\nSample Input Shape (Before FFN Block Norm):")
print(f"  input_to_ffn_block: {input_to_ffn_block.shape}")

# 2. Pre-Normalization (post-attention layernorm)

# RMSNorm Implementation
class SimplifiedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Learnable parameters
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype  # FIX: attribute, not callable
        hidden_states = hidden_states.to(torch.float32)  # Compute in float32 for stability
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # Variance over last dim
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

# Instantiate and apply normalization
post_attn_norm = SimplifiedRMSNorm(hidden_size, eps=rms_norm_eps)
normalized_hidden_states = post_attn_norm(input_to_ffn_block)

print("\nShape after Post-Attention RMSNorm:")
print(f"  normalized_hidden_states: {normalized_hidden_states.shape}")

# 3. Implementing The FeedForward Network (SwiGLU-style MLP)

# FFN layers
gate_proj = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
up_proj = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
down_proj = nn.Linear(intermediate_size, hidden_size, bias=ffn_bias)

# Activation function
if hidden_act == "silu":
    activation_fn = nn.SiLU()
else:
    raise NotImplementedError(f"Activation {hidden_act} not implemented here")

# Apply FFN layers
gate_output = gate_proj(normalized_hidden_states)
up_output = up_proj(normalized_hidden_states)

# SwiGLU: apply activation on gate, then element-wise multiply
activated_gate = activation_fn(gate_output)
gated_result = activated_gate * up_output

# Final down projection
ffn_output = down_proj(gated_result)

print("\nShapes within FFN:")
print(f"  gate_output: {gate_output.shape}")
print(f"  up_output: {up_output.shape}")
print(f"  gated_result: {gated_result.shape}")
print(f"  ffn_output: {ffn_output.shape}")

# Residual connection
final_output = input_to_ffn_block + ffn_output

print("\nShape after FFN Residual Connection:")
print(f"  final_output: {final_output.shape}")

# 4. Full LLaMA4 FFN Block as a Module
class Llama4FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.hidden_act = config["hidden_act"]
        self.ffn_bias = config["ffn_bias"]
        self.rms_norm_eps = config["rms_norm_eps"]

        # Normalization layer
        self.norm = SimplifiedRMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # MLP layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.ffn_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.ffn_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.ffn_bias)

        # Activation
        if self.hidden_act == "silu":
            self.activation_fn = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation {self.hidden_act} not implemented")

    def forward(self, hidden_states):
        # Apply pre-FFN normalization
        normalized_states = self.norm(hidden_states)

        # SwiGLU MLP
        gate = self.gate_proj(normalized_states)
        up = self.up_proj(normalized_states)
        down = self.down_proj(self.activation_fn(gate) * up)

        # Residual connection applied outside
        return down

# Instantiate and run module
ffn_config_dict = {
    'hidden_size': hidden_size,
    'intermediate_size': intermediate_size,
    'hidden_act': hidden_act,
    'ffn_bias': ffn_bias,
    'rms_norm_eps': rms_norm_eps,
}
ffn_module = Llama4FFN(ffn_config_dict)

mlp_output_from_module = ffn_module(input_to_ffn_block)
final_output_from_module = input_to_ffn_block + mlp_output_from_module

print("\nOutput shape from simplified FFN module (before residual):", mlp_output_from_module.shape)
print("Output shape after external residual connection:", final_output_from_module.shape)
print("Outputs are close:", torch.allclose(final_output, final_output_from_module, atol=1e-6))
