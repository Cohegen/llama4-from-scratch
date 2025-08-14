import torch
import torch.nn as nn
import torch.nn.functional as F

from feedForward import Llama4FFN # Experts will be Llama4FFN instances

class MoERouter(nn.Module):
    def __init__(self, hidden_size, num_experts, num_experts_per_token):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, sequence_length, hidden_size)
        
        # Compute expert logits
        router_logits = self.gate(hidden_states) # (batch_size, sequence_length, num_experts)

        # Apply softmax to get routing weights
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        # top_k_weights: (batch_size, sequence_length, num_experts_per_token)
        # top_k_indices: (batch_size, sequence_length, num_experts_per_token)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.num_experts_per_token, dim=-1)

        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return top_k_weights, top_k_indices

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_size = config.hidden_size

        self.router = MoERouter(self.hidden_size, self.num_experts, self.num_experts_per_token)
        self.experts = nn.ModuleList([
            Llama4FFN(config.to_dict()) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states):
        # hidden_states: (batch_size, sequence_length, hidden_size)
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # Flatten hidden_states for routing (each token is routed independently)
        flat_hidden_states = hidden_states.view(-1, hidden_size) # (batch_size * sequence_length, hidden_size)

        # Get routing weights and indices
        top_k_weights, top_k_indices = self.router(hidden_states)
        # Reshape top_k_weights and top_k_indices to (batch_size * sequence_length, num_experts_per_token)
        top_k_weights = top_k_weights.view(-1, self.num_experts_per_token)
        top_k_indices = top_k_indices.view(-1, self.num_experts_per_token)

        # Initialize output tensor
        output = torch.zeros_like(flat_hidden_states) # (batch_size * sequence_length, hidden_size)

        # Iterate over experts and process tokens
        for i, expert in enumerate(self.experts):
            # Create a mask for tokens assigned to this expert
            expert_mask = (top_k_indices == i)
            
            # Get indices of tokens assigned to this expert
            flat_expert_indices = torch.nonzero(expert_mask, as_tuple=True)
            
            if flat_expert_indices[0].numel() > 0: # Check if any tokens are assigned to this expert
                # Extract tokens for this expert
                tokens_for_expert = flat_hidden_states[flat_expert_indices[0]]
                
                # Process tokens through the expert
                expert_output = expert(tokens_for_expert)
                
                # Get the corresponding weights for these tokens and this expert
                weights_for_expert = top_k_weights[flat_expert_indices]
                
                # Add weighted expert output to the overall output
                output.index_add_(0, flat_expert_indices[0], expert_output * weights_for_expert.unsqueeze(-1))

        # Reshape output back to original shape
        output = output.view(batch_size, sequence_length, hidden_size)
        return output

if __name__ == "__main__":
    from config import ModelConfig

    # Test MoE Layer
    config = ModelConfig()
    config.hidden_size = 128
    config.num_experts = 4
    config.num_experts_per_token = 2

    moe_layer = MoELayer(config)
    print("MoE Layer created:", moe_layer)

    # Dummy input
    batch_size = 2
    sequence_length = 10
    hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size)

    # Forward pass
    output = moe_layer(hidden_states)
    print("Output shape from MoE Layer:", output.shape)
