class ModelConfig:
    def __init__(self):
        self.hidden_size = 128
        self.num_attention_heads = 16
        self.num_key_value_heads = 4
        self.max_position_embeddings = 256
        self.rope_theta = 10000.0
        self.attention_bias = False
        self.use_qk_norm = True
        self.ffn_intermediate_ratio = 8/3
        self.multiple_of = 32
        self.intermediate_size = self._calculate_intermediate_size()
        self.hidden_act = "silu"
        self.rms_norm_eps = 1e-5
        self.ffn_bias = False
        self.num_hidden_layers = 2 # Number of Transformer blocks
        self.vocab_size = 50000 # Placeholder, will be updated by tokenizer

        # MoE specific parameters
        self.num_experts = 4  # Total number of experts
        self.num_experts_per_token = 2 # Number of experts to activate per token

    def _calculate_intermediate_size(self):
        intermediate_size = int(self.hidden_size * self.ffn_intermediate_ratio)
        return ((intermediate_size + self.multiple_of - 1) // self.multiple_of) * self.multiple_of

    def to_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "attention_bias": self.attention_bias,
            "use_qk_norm": self.use_qk_norm,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "ffn_bias": self.ffn_bias,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": self.vocab_size,
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
        }

if __name__ == "__main__":
    config = ModelConfig()
    print("Model Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")