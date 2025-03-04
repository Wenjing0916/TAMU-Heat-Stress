import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    """Multi-head attention mechanism that handles inputs with additional dimensions."""

    def __init__(self, model_dim, num_heads=4, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # query, key, value: (batch_size, ..., length, model_dim)
        batch_size = query.size(0)
        extra_dims = query.size()[1:-2]  # Tuple of additional dimensions, if any
        length = query.size(-2)

        # Flatten extra dimensions into batch dimension for processing
        if extra_dims:
            new_batch_size = batch_size * int(torch.prod(torch.tensor(extra_dims)))
            Q = self.FC_Q(query).view(new_batch_size, length, self.model_dim)
            K = self.FC_K(key).view(new_batch_size, length, self.model_dim)
            V = self.FC_V(value).view(new_batch_size, length, self.model_dim)
        else:
            Q = self.FC_Q(query).view(batch_size, length, self.model_dim)
            K = self.FC_K(key).view(batch_size, length, self.model_dim)
            V = self.FC_V(value).view(batch_size, length, self.model_dim)

        # Split into multiple heads
        Q = Q.view(-1, length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size*, num_heads, length, head_dim)
        K = K.view(-1, length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(-1, length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5  # (batch_size*, num_heads, length, length)

        if self.mask:
            seq_length = attn_scores.size(-1)
            mask = torch.triu(torch.ones(seq_length, seq_length, device=query.device), diagonal=1).bool()
            attn_scores.masked_fill_(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size*, num_heads, length, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, length, self.model_dim)

        # Restore original batch and extra dimensions
        if extra_dims:
            attn_output = attn_output.view(batch_size, *extra_dims, length, self.model_dim)
        else:
            attn_output = attn_output.view(batch_size, length, self.model_dim)

        output = self.out_proj(attn_output)  # (batch_size, ..., length, model_dim)

        return output



class SelfAttentionLayer(nn.Module):
    """Self-attention layer with residual connection and feed-forward network."""

    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=4, dropout=0.1, mask=False):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        # x: (batch_size, length, model_dim)
        x = x.transpose(dim, -2)  # Bring the attention dimension to the second position
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)  # Restore original dimensions
        return out


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for decoder to attend to encoder outputs."""

    def __init__(self, model_dim, num_heads=4, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        # Linear projections for query, key, and value
        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(model_dim, model_dim)
        self.value_proj = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query: (batch_size * H_W, T_out, model_dim)
        # key, value: (batch_size * H_W, T_in, model_dim)
        batch_size = query.size(0)
        T_out = query.size(1)

        # Linear projections
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        # Split into multiple heads
        Q = Q.view(batch_size, T_out, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, T_out, self.model_dim)
        output = self.out_proj(attn_output)
        return output


class EncoderLayer(nn.Module):
    """Encoder layer combining spatial and temporal attention."""

    def __init__(self, model_dim, num_heads, feed_forward_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.spatial_attn = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
        self.temporal_attn = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)

    def forward(self, x):
        # Spatial attention
        x = self.spatial_attn(x, dim=2)  # Attention over spatial dimension
        # Temporal attention
        x = self.temporal_attn(x, dim=1)  # Attention over temporal dimension
        return x


class DecoderLayer(nn.Module):
    """Decoder layer with masked self-attention and cross-attention."""

    def __init__(self, model_dim, num_heads, feed_forward_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = SelfAttentionLayer(
            model_dim, feed_forward_dim, num_heads, dropout, mask=True
        )
        self.cross_attn = CrossAttentionLayer(model_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ln3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        batch_size, T_out, H_W, model_dim = x.size()
        batch_size_mem, T_in, H_W_mem, model_dim_mem = memory.size()
        assert H_W == H_W_mem, "Spatial dimensions must match between x and memory"
        assert model_dim == model_dim_mem, "Model dimensions must match"

        # Temporal self-attention with masking
        residual = x
        x = self.self_attn(x, dim=1)
        x = self.ln1(x + residual)

        # Cross-attention with encoder output
        residual = x

        # Reshape x and memory to (batch_size * H_W, T_out, model_dim)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * H_W, T_out, model_dim)
        memory_reshaped = memory.permute(0, 2, 1, 3).reshape(batch_size * H_W, T_in, model_dim)

        x = self.cross_attn(x_reshaped, memory_reshaped, memory_reshaped)

        # Reshape back to original dimensions
        x = x.reshape(batch_size, H_W, T_out, model_dim).permute(0, 2, 1, 3)
        x = self.ln2(x + residual)

        # Feed-forward network
        residual = x
        x = self.feed_forward(x)
        x = self.ln3(x + residual)

        return x

class SpatialTemporalTransformer_Decoder(nn.Module):
    """Decoder-only model with spatial and temporal attention."""

    def __init__(self, H, W, C_in, C_temp, T_in, C_out,
                 hidden_dim=64, num_heads=4, num_layers=1, dropout=0.1):
        super(SpatialTemporalTransformer_Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.C_out = C_out
        self.T_in = T_in
        self.H = H
        self.W = W
        self.H_W = H * W
        self.C_temp = C_temp  # X_temp input features
        self.C_in = C_in      # X_in input features

        # Linear layer to project C_in to hidden_dim
        self.fc_in = nn.Linear(self.C_in, self.hidden_dim)

        # Linear layer to project C_temp to hidden_dim
        self.fc_temp = nn.Linear(self.C_temp, self.hidden_dim)

        # Spatial attention layers for X_in
        self.spatial_attn_layers = nn.ModuleList([
            SelfAttentionLayer(self.hidden_dim, feed_forward_dim=hidden_dim*4, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Temporal attention layers for X_in
        self.temporal_attn_layers = nn.ModuleList([
            SelfAttentionLayer(self.hidden_dim, feed_forward_dim=hidden_dim*4, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Temporal attention layers for X_temp
        self.temp_temporal_attn_layers = nn.ModuleList([
            SelfAttentionLayer(self.hidden_dim, feed_forward_dim=hidden_dim*4, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.C_out)

    def forward(self, X_in, X_temp):
        batch_size = X_in.size(0)
        H = self.H
        W = self.W
        H_W = H * W
        T_in = self.T_in

        # Process X_in
        X_in = X_in.permute(0, 4, 1, 2, 3)  # (batch_size, T_in, H, W, C_in)
        x_in = X_in.reshape(batch_size, T_in, H_W, self.C_in)  # (batch_size, T_in, H*W, C_in)
        x_in = self.fc_in(x_in)  # (batch_size, T_in, H*W, hidden_dim)

        # Apply spatial attention
        for layer in self.spatial_attn_layers:
            x_in = layer(x_in, dim=2)

        # Apply temporal attention
        for layer in self.temporal_attn_layers:
            x_in = layer(x_in, dim=1)

        # Process X_temp
        x_temp = self.fc_temp(X_temp)  # (batch_size, T_in, hidden_dim)

        # Apply temporal attention
        for layer in self.temp_temporal_attn_layers:
            x_temp = layer(x_temp, dim=1)

        # Expand x_temp to match x_in's spatial dimensions
        x_temp_expanded = x_temp.unsqueeze(2).repeat(1, 1, H_W, 1)  # (batch_size, T_in, H*W, hidden_dim)

        # Combine x_in and x_temp
        x_combined = x_in + x_temp_expanded  # (batch_size, T_in, H*W, hidden_dim)

        # Output projection
        output = self.output_proj(x_combined)  # (batch_size, T_in, H*W, C_out)

        # Reshape to (batch_size, H, W, C_out, T_in)
        output = output.view(batch_size, T_in, H, W, self.C_out)
        output = output.permute(0, 2, 3, 4, 1)  # (batch_size, H, W, C_out, T_in)

        return output

class SpatialTemporalTransformer(nn.Module):
    """Full Transformer model with spatial and temporal attention."""

    def __init__(self, H=256, W=256, C_in=4, C_temp=7, T_in=24, T_out=24, C_out=1,
                 hidden_dim=64, num_heads=4, num_layers=1, dropout=0.1):
        super(SpatialTemporalTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.C_out = C_out
        self.T_in = T_in
        self.T_out = T_out
        self.H = H
        self.W = W
        self.H_W = H * W
        self.C_temp = C_temp  # X_temp input features
        self.C_in = C_in      # X_in input features

        # Linear layer to compress C_in dimension of X_in
        self.fc = nn.Linear(self.C_in, self.hidden_dim)

        # Linear layer to project X_temp to hidden_dim
        self.temp_fc = nn.Linear(self.C_temp, self.hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.hidden_dim, num_heads, hidden_dim * 4, dropout) for _ in range(num_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(self.hidden_dim, num_heads, hidden_dim * 4, dropout) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.C_out)

    def forward(self, X_in, X_temp):
        batch_size = X_in.size(0)
        H = self.H
        W = self.W
        H_W = H * W
        T_in = self.T_in
        T_out = self.T_out

        # Process X_in
        X_in = X_in.permute(0, 4, 1, 2, 3)  # (batch_size, T_in, H, W, C_in)
        x = X_in.reshape(batch_size, T_in, H_W, self.C_in)  # (batch_size, T_in, H*W, C_in)
        x = self.fc(x)  # (batch_size, T_in, H*W, hidden_dim)

        # Process X_temp
        X_temp = self.temp_fc(X_temp)  # (batch_size, T_in, hidden_dim)
        X_temp = X_temp.unsqueeze(2).repeat(1, 1, H_W, 1)  # (batch_size, T_in, H*W, hidden_dim)

        # Combine X_in and X_temp
        encoder_input = x + X_temp  # (batch_size, T_in, H*W, hidden_dim)

        # Encoder
        for layer in self.encoder_layers:
            encoder_input = layer(encoder_input)

        # Prepare decoder input (start tokens or zeros)
        decoder_input = torch.zeros(batch_size, T_out, H_W, self.hidden_dim, device=X_in.device)

        # Decoder
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input, encoder_input)

        # Output projection
        output = self.output_proj(decoder_input)  # (batch_size, T_out, H*W, C_out)
        output = output.reshape(batch_size, T_out, H, W, self.C_out)
        output = output.permute(0, 2, 3, 4, 1)  # (batch_size, H, W, C_out, T_out)

        return output
    
    
    