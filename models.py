import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, Sequential, Linear, ReLU
import math
from copy import deepcopy

def sinusoidal_positional_encoding(seq_len, d_model, device):
    """Standard sinusoidal PE (Vaswani et al., 2017)."""
    position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)
# end sinusoidal_positional_encoding

# ========== SINGLE ENCODER MODEL ==========

class TransformerEncoderLayerWithAttn(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None  # place to store the weights

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # same as parent forward, except we intercept attn_weights
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        if not self.training:
            self.last_attn_weights = attn_weights.detach()  # store for later

        # rest of the computation is copied from TransformerEncoderLayer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
# end TransformerEncoderLayerWithAttn

# Modular single encoder
class SEModel(nn.Module):
    def __init__(
            self, 
            chord_vocab_size,  # V
            d_model=512, 
            nhead=8, 
            num_layers=8, 
            dim_feedforward=2048,
            pianoroll_dim=13,      # PCP + bars only
            grid_length=80,
            dropout=0.3,
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length

        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)

        self.seq_len = grid_length + grid_length
        
        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            grid_length, d_model, device
        )
        self.full_pos = torch.cat([self.shared_pos[:, :self.grid_length, :],
                            self.shared_pos[:, :self.grid_length, :]], dim=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=num_layers)
        # Optional: output head for harmonies
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=self.device)
        # Layer norm at input and output
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.to(device)
    # end init

    def forward(self, melody_grid, harmony_tokens=None):
        """
        melody_grid: (B, grid_length, pianoroll_dim)
        harmony_tokens: (B, grid_length) - optional for training or inference
        """
        B = melody_grid.size(0)
        device = self.device

        # Project melody: (B, grid_length, pianoroll_dim) → (B, grid_length, d_model)
        melody_emb = self.melody_proj(melody_grid)

        # Harmony token embedding (optional for training): (B, grid_length) → (B, grid_length, d_model)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Placeholder (zeros) if not provided
            harmony_emb = torch.zeros(B, self.grid_length, self.d_model, device=device)

        # Concatenate full input: (B, 1 + grid_length + grid_length, d_model)
        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.full_pos

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        # Transformer encode
        encoded = self.encoder(full_seq)
        encoded = self.output_norm(encoded)

        # Optionally decode harmony logits (only last grid_length tokens)
        harmony_output = self.output_head(encoded[:, -self.grid_length:, :])  # (B, grid_length, V)

        return harmony_output
    # end forward

    def freeze_base(self):
        pass
    def freeze_FiLM(self):
        pass
    def unfreeze_all(self):
        pass

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder.layers:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end class SEModel

# ================ Activation steering =======================

# Modular single encoder
class SEASModel(nn.Module):
    def __init__(
            self, 
            chord_vocab_size,  # V
            d_model=512, 
            nhead=8, 
            num_layers=8, 
            dim_feedforward=2048,
            pianoroll_dim=13,      # PCP + bars only
            grid_length=80,
            dropout=0.3,
            guidance_dim=None,
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length

        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)

        self.seq_len = grid_length + grid_length
        
        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            grid_length, d_model, device
        )
        self.full_pos = torch.cat([self.shared_pos[:, :self.grid_length, :],
                            self.shared_pos[:, :self.grid_length, :]], dim=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder
        # encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, 
        #                                            nhead=nhead, 
        #                                            dim_feedforward=dim_feedforward,
        #                                            dropout=dropout,
        #                                            activation='gelu',
        #                                            batch_first=True)
        # self.encoder = nn.TransformerEncoder(
        #                 encoder_layer,
        #                 num_layers=num_layers)
        self.encoder_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for _ in range(num_layers):
            base_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.encoder_layers.append(
                TransformerEncoderLayerWithFiLM(
                    base_layer,
                    d_model=d_model,
                    guidance_dim=guidance_dim
                )
            )
            self.film_layers.append(self.encoder_layers[-1].film)
        # Optional: output head for harmonies
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=self.device)
        # Layer norm at input and output
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.to(device)
    # end init

    def forward(
            self,
            melody_grid,
            harmony_tokens=None,
            steering_vectors=None,
            alpha=1.0,
            get_layers_output=False
        ):
        """
        melody_grid: (B, grid_length, pianoroll_dim)
        harmony_tokens: (B, grid_length) - optional for training or inference
        """
        B = melody_grid.size(0)
        device = self.device

        # Project melody: (B, grid_length, pianoroll_dim) → (B, grid_length, d_model)
        melody_emb = self.melody_proj(melody_grid)

        # Harmony token embedding (optional for training): (B, grid_length) → (B, grid_length, d_model)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Placeholder (zeros) if not provided
            harmony_emb = torch.zeros(B, self.grid_length, self.d_model, device=device)

        # Concatenate full input: (B, 1 + grid_length + grid_length, d_model)
        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.full_pos

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        if get_layers_output:
            layers_output = {}
        # Transformer encode
        encoded = full_seq
        for i, layer in enumerate(self.encoder_layers):
            encoded = layer(encoded)
            # capture layer output
            if get_layers_output:
                layers_output[i] = encoded[:, -self.grid_length:, :].mean(axis=1).unsqueeze(1)
            # guide process
            if steering_vectors is not None and i in steering_vectors:
                h = steering_vectors[i]
                # ensure correct shape
                if h.dim() == 2:  # (1, d_model)
                    h = h.unsqueeze(1)  # (1,1,d_model)
                encoded[:, -self.grid_length:, :] += alpha * h
        encoded = self.output_norm(encoded)

        # Optionally decode harmony logits (only last grid_length tokens)
        harmony_output = self.output_head(encoded[:, -self.grid_length:, :])  # (B, grid_length, V)
        
        if get_layers_output:
            return harmony_output, layers_output
        else:
            return harmony_output
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder.layers:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end class SEASModel

# ======================== FiLM ==============================

class FiLMAdapter(nn.Module):
    def __init__(self, d_model, guidance_dim):
        super().__init__()
        self.gamma = nn.Linear(guidance_dim, d_model)
        self.beta = nn.Linear(guidance_dim, d_model)

        # neutralize FiLM
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
    # end init

    def forward(self, x, guidance):
        if guidance is None:
            return x

        gamma = self.gamma(guidance).unsqueeze(1)
        beta = self.beta(guidance).unsqueeze(1)
        
        return gamma * x + beta
    # end forward
# end FiLMAdapter

class TransformerEncoderLayerWithFiLM(nn.Module):
    def __init__(self, encoder_layer, d_model, guidance_dim=None):
        super().__init__()
        self.layer = encoder_layer
        self.guidance_dim = guidance_dim

        self.film = FiLMAdapter(d_model, guidance_dim)

        self.last_attn_weights = None
    # end init

    def forward(self, src, guidance=None, src_mask=None, src_key_padding_mask=None):
        # --- Standard TransformerEncoderLayer forward ---
        src2, attn_weights = self.layer.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )

        if not self.training:
            self.last_attn_weights = attn_weights.detach()

        src = src + self.layer.dropout1(src2)
        src = self.layer.norm1(src)
        src2 = self.layer.linear2(
            self.layer.dropout(
                self.layer.activation(self.layer.linear1(src))
            )
        )
        src = src + self.layer.dropout2(src2)
        src = self.layer.norm2(src)

        # --- FiLM conditioning ---
        if guidance is not None:
            src = self.film(src, guidance)

        return src
    # end forward
# end TransformerEncoderLayerWithFiLM

class SEFiLMModel(nn.Module):
    def __init__(
        self,
        chord_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        pianoroll_dim=13,
        grid_length=80,
        dropout=0.3,
        guidance_dim=None,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length
        self.guidance_dim = guidance_dim

        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)

        shared = sinusoidal_positional_encoding(self.grid_length, d_model, device)
        self.shared_pos = torch.cat([shared, shared], dim=1)

        self.dropout = nn.Dropout(dropout)

        # --- Encoder Layers with FiLM ---
        self.encoder_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        for _ in range(num_layers):
            base_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.encoder_layers.append(
                TransformerEncoderLayerWithFiLM(
                    base_layer,
                    d_model=d_model,
                    guidance_dim=guidance_dim
                )
            )
            self.film_layers.append(self.encoder_layers[-1].film)

        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)

        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self.to(device)
    # end init

    def forward(
        self,
        melody_grid,
        harmony_tokens=None,
        guidance_embedding=None,
        return_hidden=False
    ):
        B = melody_grid.size(0)
        device = self.device

        melody_emb = self.melody_proj(melody_grid)

        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            harmony_emb = torch.zeros(
                B, self.grid_length, self.d_model, device=device
            )

        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)

        full_seq = full_seq + self.shared_pos
        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        encoded = full_seq

        for layer in self.encoder_layers:
            encoded = layer(encoded, guidance=guidance_embedding)

        encoded = self.output_norm(encoded)

        harmony_logits = self.output_head(
            encoded[:, -self.grid_length:, :]
        )

        if return_hidden:
            return harmony_logits, torch.mean(encoded[:, -self.grid_length:, :], axis=1).squeeze()
        else:
            return harmony_logits
    # end forward

    def freeze_base(self):
        for param in self.parameters():
            param.requires_grad = False
        for m in self.film_layers:
            for param in m.parameters():
                param.requires_grad = True
    # end freeze_base

    def freeze_FiLM(self):
        for param in self.parameters():
            param.requires_grad = True
        for m in self.film_layers:
            for param in m.parameters():
                param.requires_grad = False
    # end freeze_FiLM

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    # end unfreeze_all

    def film_parameters(self):
        for m in self.film_layers:
            yield from m.parameters()
    # end film_parameters
# end SEFiLMModel


# ====================== Encoder-decoder =========================

class TransformerDecoderLayerWithFiLM(nn.Module):
    def __init__(self, decoder_layer, d_model, guidance_dim=None):
        super().__init__()
        self.layer = decoder_layer
        self.guidance_dim = guidance_dim

        self.film = FiLMAdapter(d_model, guidance_dim)

        self.last_attn_weights = None
    # end init

    def forward(self, src, enc_output, guidance=None, src_mask=None, src_key_padding_mask=None):
        # --- Standard TransformerDecoderLayer forward ---
        # --- Self-attention ---
        src2 = self.layer.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]

        src = src + self.layer.dropout1(src2)
        src = self.layer.norm1(src)

        # --- Cross-attention ---
        src2, attn_weights = self.layer.multihead_attn(
            src, enc_output, enc_output,
            need_weights=True,
            average_attn_weights=False
        )

        if not self.training:
            self.last_attn_weights = attn_weights.detach()

        src = src + self.layer.dropout1(src2)
        src = self.layer.norm1(src)
        src2 = self.layer.linear2(
            self.layer.dropout(
                self.layer.activation(self.layer.linear1(src))
            )
        )
        src = src + self.layer.dropout2(src2)
        src = self.layer.norm2(src)

        # --- FiLM conditioning ---
        if guidance is not None:
            src = self.film(src, guidance)

        return src
    # end forward
# end TransformerEncoderLayerWithFiLM

class EDFiLMModel(nn.Module):
    def __init__(
        self,
        chord_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        pianoroll_dim=13,
        grid_length=80,
        dropout=0.3,
        guidance_dim=None,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length
        self.guidance_dim = guidance_dim

        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)

        self.shared_pos = sinusoidal_positional_encoding(self.grid_length, d_model, device)
        # self.shared_pos = torch.cat([shared, shared], dim=1)

        self.dropout = nn.Dropout(dropout)

        # --- Encoder Layers ---
        self.encoder_layers = nn.ModuleList()
        # --- Decoder Layers with FiLM ---
        self.decoder_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.tgt_mask = self.generate_square_subsequent_mask(self.grid_length).to(device)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=num_layers)

        for _ in range(num_layers):
            base_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.decoder_layers.append(
                TransformerDecoderLayerWithFiLM(
                    base_layer,
                    d_model=d_model,
                    guidance_dim=guidance_dim
                )
            )
            self.film_layers.append(self.decoder_layers[-1].film)

        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)

        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self.to(device)
    # end init

    def forward(
        self,
        melody_grid,
        harmony_tokens=None,
        guidance_embedding=None,
        return_hidden=False
    ):
        B = melody_grid.size(0)
        device = self.device

        melody_emb = self.melody_proj(melody_grid)

        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            harmony_emb = torch.zeros(
                B, self.grid_length, self.d_model, device=device
            )

        # melody to encoder
        melody_encoded = self.encoder(melody_emb)
        melody_encoded = self.output_norm(melody_encoded)

        # harmony to decoder
        harmony_emb = harmony_emb + self.shared_pos
        harmony_emb = self.input_norm(harmony_emb)
        harmony_emb = self.dropout(harmony_emb)
        encoded = harmony_emb
        for layer in self.decoder_layers:
            encoded = layer(
                encoded,
                melody_encoded,
                guidance=guidance_embedding,
                src_mask=self.tgt_mask
            )
        encoded = self.output_norm(encoded)

        harmony_logits = self.output_head(
            encoded
        )

        if return_hidden:
            return harmony_logits, torch.mean(encoded, axis=1).squeeze()
        else:
            return harmony_logits
    # end forward

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    # end generate_square_subsequent_mask

    def freeze_base(self):
        for param in self.parameters():
            param.requires_grad = False
        for m in self.film_layers:
            for param in m.parameters():
                param.requires_grad = True
    # end freeze_base

    def freeze_FiLM(self):
        for param in self.parameters():
            param.requires_grad = True
        for m in self.film_layers:
            for param in m.parameters():
                param.requires_grad = False
    # end freeze_FiLM

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    # end unfreeze_all

    def film_parameters(self):
        for m in self.film_layers:
            yield from m.parameters()
    # end film_parameters
# end EDFiLMModel

# ============== Activation Steering ==================

class EDASModel(nn.Module):
    def __init__(
        self,
        chord_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        pianoroll_dim=13,
        grid_length=80,
        dropout=0.3,
        guidance_dim=None,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length
        self.guidance_dim = guidance_dim

        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)

        self.shared_pos = sinusoidal_positional_encoding(self.grid_length, d_model, device)
        # self.shared_pos = torch.cat([shared, shared], dim=1)

        self.dropout = nn.Dropout(dropout)

        # --- Encoder Layers ---
        self.encoder_layers = nn.ModuleList()
        # --- Decoder Layers with FiLM ---
        self.decoder_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.tgt_mask = self.generate_square_subsequent_mask(self.grid_length).to(device)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=num_layers)

        for _ in range(num_layers):
            base_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.decoder_layers.append(
                TransformerDecoderLayerWithFiLM(
                    base_layer,
                    d_model=d_model,
                    guidance_dim=guidance_dim
                )
            )
            self.film_layers.append(self.decoder_layers[-1].film)

        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)

        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self.to(device)
    # end init

    def forward(
            self,
            melody_grid,
            harmony_tokens=None,
            steering_vectors=None,
            alpha=1.0,
            get_layers_output=False
        ):
        B = melody_grid.size(0)
        device = self.device

        melody_emb = self.melody_proj(melody_grid)

        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            harmony_emb = torch.zeros(
                B, self.grid_length, self.d_model, device=device
            )

        # melody to encoder
        melody_encoded = self.encoder(melody_emb)
        melody_encoded = self.output_norm(melody_encoded)

        # harmony to decoder
        harmony_emb = harmony_emb + self.shared_pos
        harmony_emb = self.input_norm(harmony_emb)
        harmony_emb = self.dropout(harmony_emb)
        encoded = harmony_emb
        
        if get_layers_output:
            layers_output = {}
        for i, layer in enumerate(self.decoder_layers):
            encoded = layer(
                encoded,
                melody_encoded,
                src_mask=self.tgt_mask
            )
            # capture layer output
            if get_layers_output:
                layers_output[i] = encoded.mean(axis=1).unsqueeze(1)
            # guide process
            if steering_vectors is not None and i in steering_vectors:
                h = steering_vectors[i]
                # ensure correct shape
                if h.dim() == 2:  # (1, d_model)
                    h = h.unsqueeze(1)  # (1,1,d_model)
                encoded += alpha * h
        encoded = self.output_norm(encoded)

        harmony_logits = self.output_head(
            encoded
        )

        if get_layers_output:
            return harmony_logits, layers_output
        else:
            return harmony_logits
    # end forward

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    # end generate_square_subsequent_mask

    # def freeze_base(self):
    #     for param in self.parameters():
    #         param.requires_grad = False
    #     for m in self.film_layers:
    #         for param in m.parameters():
    #             param.requires_grad = True
    # # end freeze_base

    # def freeze_FiLM(self):
    #     for param in self.parameters():
    #         param.requires_grad = True
    #     for m in self.film_layers:
    #         for param in m.parameters():
    #             param.requires_grad = False
    # # end freeze_FiLM

    # def unfreeze_all(self):
    #     for param in self.parameters():
    #         param.requires_grad = True
    # # end unfreeze_all

    # def film_parameters(self):
    #     for m in self.film_layers:
    #         yield from m.parameters()
    # # end film_parameters
# end EDASModel