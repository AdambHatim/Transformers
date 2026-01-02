import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def Positionnal_Encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_attention, num_heads, dropout):
        super().__init__()
        self.d_attention = d_attention
        self.num_heads = num_heads


        assert d_model == num_heads * d_attention


        self.Q = nn.Linear(d_model, num_heads * d_attention, bias=False)
        self.K = nn.Linear(d_model, num_heads * d_attention, bias=False)
        self.V = nn.Linear(d_model, num_heads * d_attention, bias=False)


        self.out_proj = nn.Linear(num_heads * d_attention, d_model)
        self.attn_dropout = nn.Dropout(dropout)

   

    # queries shape (Batch_size, q_len, d_model)
    # keys shape (Batch_size, k_len, d_model)
    # values shape (Batch_size, v_len, d_model)
    # pad_mask shape (Batch_size, k_len)
    def forward(self, queries, keys, values, causal_mask=False, pad_mask=None):
        B, q_len, _ = queries.shape
        _, k_len, _ = keys.shape


        Q = self.Q(queries)
        K = self.K(keys)
        V = self.V(values)


        Q = Q.reshape(B, q_len, self.num_heads, self.d_attention).transpose(1, 2)
        K = K.reshape(B, k_len, self.num_heads, self.d_attention).transpose(1, 2)
        V = V.reshape(B, k_len, self.num_heads, self.d_attention).transpose(1, 2)


        scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.d_attention)


        if pad_mask is not None:
            mask = pad_mask.bool().unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))


        if causal_mask:
            causal = torch.triu(
                torch.ones(q_len, k_len, device=scores.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))


        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)


        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, q_len, self.num_heads * self.d_attention)


        return self.out_proj(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_attention, d_ffn, num_heads, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, d_attention, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn1 = nn.Linear(d_model, d_ffn)
        self.ffn2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_pad_mask=None):
        x = x + self.dropout(
            self.attn(self.norm1(x), self.norm1(x), self.norm1(x), pad_mask=src_pad_mask)
        )
        x = x + self.dropout(
            self.ffn2(F.relu(self.ffn1(self.norm2(x))))
        )
        return x


class Encoder(nn.Module):
  def __init__(self, num_encoder_layers, d_model, d_attention, num_heads, d_ffn, dropout):
    super().__init__()


    self.layers = nn.ModuleList([EncoderLayer(d_model, d_attention, d_ffn, num_heads, dropout)
                                  for _ in range(num_encoder_layers)])
  def forward(self, x, src_pad_mask = None):
     for layer in self.layers:
       x = layer(x, src_pad_mask)
     return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_attention, num_heads, d_ffn, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, d_attention, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)


        self.cross_attn = MultiHeadAttention(d_model, d_attention, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)


        self.ffn1 = nn.Linear(d_model, d_ffn)
        self.ffn2 = nn.Linear(d_ffn, d_model)
        self.norm3 = nn.LayerNorm(d_model)


        self.dropout = nn.Dropout(dropout)


    def forward(self, x, x_enc, tgt_pad_mask=None, src_pad_mask=None):
        # Pre-LN masked self-attention
        x = x + self.dropout(
            self.self_attn(
                self.norm1(x), self.norm1(x), self.norm1(x),
                causal_mask=True, pad_mask=tgt_pad_mask
            )
        )
        # Pre-LN cross-attention
        x = x + self.dropout(
            self.cross_attn(
                self.norm2(x), x_enc, x_enc,
                pad_mask=src_pad_mask
            )
        )
        # Pre-LN FFN
        x = x + self.dropout(
            self.ffn2(F.relu(self.ffn1(self.norm3(x))))
        )
        return x

class Decoder(nn.Module):
  def __init__(self, num_decoder_layers, d_model, d_attention, num_heads, d_ffn, dropout):
    super().__init__()


    self.layers = nn.ModuleList([DecoderLayer(d_model, d_attention, num_heads, d_ffn, dropout)
                                  for _ in range(num_decoder_layers)])
  def forward(self, tgt, memory, tgt_pad_mask=None, src_pad_mask=None):
      for layer in self.layers:
          tgt = layer(tgt, memory, tgt_pad_mask, src_pad_mask)
      return tgt

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, seq_length,
                 num_encoder_layers, num_decoder_layers,
                 d_model, d_attention, num_heads, d_ffn, dropout, pad_idx):
        super().__init__()


        self.d_model = d_model
        self.pad_idx = pad_idx


        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)


        self.register_buffer(
            "pos_emb",
            Positionnal_Encoding(seq_length, d_model).unsqueeze(0)
        )


        self.dropout = nn.Dropout(dropout)


        self.encoder = Encoder(num_encoder_layers, d_model, d_attention, num_heads, d_ffn, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, d_attention, num_heads, d_ffn, dropout)


        self.output_layer = nn.Linear(d_model, tgt_vocab_size)


    def forward(self, src, tgt):
        src_pad_mask = (src == self.pad_idx)
        tgt_pad_mask = (tgt == self.pad_idx)


        src = self.src_emb(src) * math.sqrt(self.d_model)
        src = src + self.pos_emb[:, :src.size(1), :]
        src = self.dropout(src)


        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt = tgt + self.pos_emb[:, :tgt.size(1), :]
        tgt = self.dropout(tgt)


        memory = self.encoder(src, src_pad_mask)
        out = self.decoder(tgt, memory, tgt_pad_mask, src_pad_mask)


        return self.output_layer(out)