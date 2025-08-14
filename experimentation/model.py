import torch
import torch.nn as nn

from attention import SimplifiedLlama4Attention
from feedForward import Llama4FFN, SimplifiedRMSNorm
from config import ModelConfig
from moe import MoELayer # Import the MoELayer

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = SimplifiedLlama4Attention(config.to_dict())
        
        # Replace Llama4FFN with MoELayer
        self.moe_layer = MoELayer(config) 
        
        self.attention_norm = SimplifiedRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = SimplifiedRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_ids):
        # Attention block
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attn_output, _ = self.attention(hidden_states, attention_mask, position_ids)
        hidden_states = residual + attn_output

        # Feed Forward (MoE) block
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        # Use moe_layer instead of feed_forward
        moe_output = self.moe_layer(hidden_states)
        hidden_states = residual + moe_output

        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = SimplifiedRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Initialize weights (simple example, typically more complex)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        # Input embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # If position_ids are not provided, create them causally
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Transformer blocks
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        # Final normalization and language model head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

if __name__ == "__main__":
    # Example usage
    config = ModelConfig()
    # For demonstration, let's set a dummy vocab_size
    config.vocab_size = 1000
    
    model = LlamaModel(config)
    print("Model created:", model)

    # Create dummy input
    batch_size = 2
    sequence_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
    
    # Create a causal attention mask
    attention_mask = torch.triu(torch.ones(sequence_length, sequence_length) * -torch.inf, diagonal=1)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    # Forward pass
    output_logits = model(input_ids, attention_mask=attention_mask)
    print("Output logits shape:", output_logits.shape)