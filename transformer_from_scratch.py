import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Weight matrices for Q, K, V projections
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        
        # Output projection matrix
        self.W_o = np.random.randn(d_model, d_model)

    def _softmax(self, x):
        # Numerically stable softmax
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute attention scores
        attention_scores = np.matmul(Q, K.T) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = np.where(mask == 0, float('-inf'), attention_scores)
        
        # Softmax attention weights
        attention_weights = self._softmax(attention_scores)
        
        # Compute output
        output = np.matmul(attention_weights, V)
        return output
    
    
    def forward(self, X, mask=None):
        batch_size, seq_length, _ = X.shape
        
        # Linear projections
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)
        
        # Split into multiple heads
        Q = Q.reshape(batch_size, seq_length, self.num_heads, self.d_k)
        K = K.reshape(batch_size, seq_length, self.num_heads, self.d_k)
        V = V.reshape(batch_size, seq_length, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))
        
        # Compute attention for each head
        head_outputs = []
        for i in range(self.num_heads):
            head_output = self.scaled_dot_product_attention(
                Q[:, i], K[:, i], V[:, i], mask
            )
            head_outputs.append(head_output)
        
        # Concatenate heads
        multi_head_output = np.concatenate(head_outputs, axis=-1)
        
        # Final linear projection
        output = np.matmul(multi_head_output, self.W_o)
        
        return output

class PositionalEncoding:
    def __init__(self, d_model, max_seq_length=5000):
        # Create positional encoding matrix
        position = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        self.pe = np.zeros((max_seq_length, d_model))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)
    
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:x.shape[1], :]

class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        # Weights for two linear transformations
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        # First linear transformation with ReLU
        x = np.maximum(0, np.matmul(x, self.W1) + self.b1)
        
        # Second linear transformation
        x = np.matmul(x, self.W2) + self.b2
        
        return x

class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        
        # Layer normalization
        self.layer_norm1 = self._layer_norm
        self.layer_norm2 = self._layer_norm
    
    def _layer_norm(self, x, epsilon=1e-5):
        # Layer normalization
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + epsilon)
    
    def forward(self, x, mask=None):
        # Multi-head attention sub-layer with residual connection
        attn_output = self.multi_head_attention.forward(x, mask)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward sub-layer with residual connection
        ff_output = self.feed_forward.forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x

class TransformerEncoder:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, d_model)
        
        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ]
    
    def forward(self, x, mask=None):
        # Embedding
        x = np.take(self.embedding, x, axis=0)
        
        # Add positional encoding
        x = self.positional_encoding.forward(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer.forward(x, mask)
        
        return x

    
class TransformerDecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        # Self-attention layer
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        # Cross-attention layer (for attending to encoder outputs)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        
        # Layer normalization
        self.layer_norm1 = self._layer_norm
        self.layer_norm2 = self._layer_norm
        self.layer_norm3 = self._layer_norm
    
    def _layer_norm(self, x, epsilon=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + epsilon)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with target mask
        self_attn_output = self.self_attention.forward(x, tgt_mask)
        x = self.layer_norm1(x + self_attn_output)
        
        # Cross-attention with encoder output
        cross_attn_output = self.cross_attention.forward(x, enc_output, src_mask)
        x = self.layer_norm2(x + cross_attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward.forward(x)
        x = self.layer_norm3(x + ff_output)
        
        return x

class TransformerDecoder:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, d_model)
        
        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Output linear layer (pre-softmax)
        self.output_layer = np.random.randn(d_model, vocab_size)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Embedding
        x = np.take(self.embedding, x, axis=0)
        
        # Add positional encoding
        x = self.positional_encoding.forward(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer.forward(x, enc_output, src_mask, tgt_mask)
        
        # Project to vocabulary size
        output = np.matmul(x, self.output_layer)
        
        return output

class FullTransformer:
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048):
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff)
        self.decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    def create_masks(self, src, tgt):
        # Source mask (for padding)
        src_mask = (src != 0).astype(np.float32)
        
        # Target mask (combination of padding and look-ahead mask)
        tgt_mask = (tgt != 0).astype(np.float32)
        seq_length = tgt.shape[1]
        
        # Create look-ahead mask
        look_ahead_mask = np.triu(np.ones((seq_length, seq_length)), k=1)
        look_ahead_mask = (1 - look_ahead_mask).astype(np.float32)
        
        # Combine padding and look-ahead masks
        tgt_mask = tgt_mask[:, np.newaxis, :] * look_ahead_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        # Create masks
        src_mask, tgt_mask = self.create_masks(src, tgt)
        
        # Encoder
        enc_output = self.encoder.forward(src, src_mask)
        
        # Decoder
        dec_output = self.decoder.forward(tgt, enc_output, src_mask, tgt_mask)
        
        return dec_output