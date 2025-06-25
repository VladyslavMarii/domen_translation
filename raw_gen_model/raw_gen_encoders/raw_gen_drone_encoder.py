import tf_keras
from tf_keras.saving import register_keras_serializable


@register_keras_serializable()
class DroneEncoder(tf_keras.Model):
    """
    Translates 'drone tokens' into 'satellite-like' tokens via a dedicated Transformer block.
    This model now accepts only drone images as input.
    """
    def __init__(self, num_heads=8, patch_size=40,input_shape=(400, 400, 3)):
        super().__init__()
        self.patch_size = patch_size
        self.dim = self.patch_size * self.patch_size * input_shape[-1]

        # Transformer block components:
        self.self_attention = tf_keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.dim)
        self.ffn = tf_keras.Sequential([
            tf_keras.layers.Dense(self.dim * 4, activation='relu'),
            tf_keras.layers.Dense(self.dim)
        ])
        
        # Dropout layers for regularization.
        # self.dropout1 = tf_keras.layers.Dropout(0.5)
        # self.dropout2 = tf_keras.layers.Dropout(0.5)
        
        # Layer normalization layers.
        self.norm1 = tf_keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf_keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, drone_tokens, training=True):
        tokens_transformed, attn_score = self.transform_tokens(drone_tokens, training=training)
        return tokens_transformed, attn_score
    
    
    def transform_tokens(self, tokens, training=True):
        """Apply a Transformer block to translate drone tokens to satellite-like tokens."""
        # Self-attention: each token attends to all tokens.
        attn_output, attn_score = self.self_attention(query=tokens, key=tokens, value=tokens, training=training, return_attention_scores=True)
        #attn_output = self.dropout1(attn_output, training=training)
        # Residual connection + LayerNorm.
        x = self.norm1(tokens + attn_output)
        
        # Feed-forward network applied to each token.
        ffn_output = self.ffn(x)
        #ffn_output = self.dropout2(ffn_output, training=training)
        # Another residual connection + LayerNorm.
        tokens_transformed = self.norm2(x + ffn_output)
        return tokens_transformed, attn_score
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)