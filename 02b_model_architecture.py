import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelConfig:
    MODEL_TYPE = "LSTM_attention"
    ATTENTION_TYPE = "MultiHead-Bahdanau-Cross-Pool"
    TOKENIZER_TYPE = "Whitespace"
    # Embedding
    LAYER_NORM_EMB = False
    FREEZE_TOKEN_EMBEDDING = True
    TOKEN_EMBEDDING = "gloVe-6B-100d"
    EMBEDDING_TYPE = "GloVe-6B-100d"
    ATTENTION_SCORE_DIM = 64  # additive score bottleneck per cross-attention head
    EMB_DIM = 100
    EMB_DP = 0.0
    # Model
    LOSS = "BCE with Logits"
    NUM_HEADS = 4
    BIDIRECTIONAL = True
    DROPOUT = 0.35
    HIDDEN_DIM = 384
    LSTM_OUT = HIDDEN_DIM*(2 if BIDIRECTIONAL else 1)
    ATTENTION_DROPOUT = 0.0
    LAYER_NORM_LSTM = False
    LAYER_NORM_ATTENTION = False
    ATTENTION_PROJECTION = False
    if ATTENTION_PROJECTION:
        PROJECT_DIM = HIDDEN_DIM // 2
    ENC_DIM = PROJECT_DIM if ATTENTION_PROJECTION else LSTM_OUT
    if LOSS == "Contrastive Loss":
        MARGIN = 1.0
    elif LOSS == "BCE with Logits":
        LABEL_SMOOTHING = 0.05
        FC_DIMS = [1024, 256]
        FC_DP = 0.4
        SIAMESE_SIMILARITY_PARM = ["Encoded Q1", "Encoded Q2", "Multiplication Q1, Q2", "Abs Subtract Q1, Q2", "Cosine Similarity"]
        MULTIPLE_FC_PARAM = sum(1 for param in SIAMESE_SIMILARITY_PARM
                     if "Q1" in param or "Q2" in param)
        INPUT_FC_DIM = MULTIPLE_FC_PARAM * ENC_DIM
        if any("Cosine" in param for param in SIAMESE_SIMILARITY_PARM):
            INPUT_FC_DIM += 1

    NUM_LAYERS = 2
    SKIP_CONNECTION = True  # cross-attention residual; not original independent pooling
    @classmethod
    def to_dict(cls):
        return {
            k.lower(): v for k, v in cls.__dict__.items()
            if not k.startswith("_")
            and not inspect.isroutine(v)   # functions, methods
            and not isinstance(v, (classmethod, staticmethod))
        }

class TokenConfig:
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    PAD_IDX = 0
    UNK_IDX = 1
    LOWERCASE = True
    UPPERCASE = False
    MAX_LENGTH = 50
    VOCAB_SIZE = 20000
    @classmethod
    def to_dict(cls):
        return {
            k.lower(): v for k, v in cls.__dict__.items()
            if not k.startswith("_")
            and not inspect.isroutine(v)   # functions, methods
            and not isinstance(v, (classmethod, staticmethod))
        }

token_cfg = TokenConfig()
model_cfg = ModelConfig()


class AdditiveCrossAttentionHead(nn.Module):
    """Token-level additive (Bahdanau) alignment: queries from one question, keys/values from the other."""
    def __init__(self, hidden_dim, value_dim, score_dim=model_cfg.ATTENTION_SCORE_DIM, dropout=0.0):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, score_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, score_dim, bias=False)
        self.score = nn.Linear(score_dim, 1, bias=False)
        self.value_proj = nn.Linear(hidden_dim, value_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, query_mask, key_mask):
        q = self.query_proj(query).unsqueeze(2)
        k = self.key_proj(key_value).unsqueeze(1)
    
        logits = self.score(torch.tanh(q + k)).squeeze(-1)
    
        # Prevent attending to padded key/value positions.
        logits = logits.masked_fill(
            ~key_mask.unsqueeze(1),
            torch.finfo(logits.dtype).min
        )
    
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
    
        # Combine the value vectors using attention weights.
        aligned = torch.bmm(weights, self.value_proj(key_value))
    
        # Zero outputs belonging to padded query positions.
        return aligned * query_mask.unsqueeze(-1).to(aligned.dtype)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=model_cfg.NUM_HEADS,
                 score_dim=model_cfg.ATTENTION_SCORE_DIM,
                 dropout=model_cfg.ATTENTION_DROPOUT):
        super().__init__()
        
        value_dim = hidden_dim // num_heads
        self.heads = nn.ModuleList([
            AdditiveCrossAttentionHead(
                hidden_dim, value_dim, score_dim, dropout
            )
            for _ in range(num_heads)
        ])

        self.out_linear = nn.Linear(value_dim*num_heads, hidden_dim)

    def forward(self, query, key_value, query_mask, key_mask):
        contexts = [
            head(query, key_value, query_mask, key_mask)
            for head in self.heads
        ]

        x = torch.cat(contexts, dim=-1)
        x = self.out_linear(x)

        return x * query_mask.unsqueeze(-1).to(x.dtype)


class AttentionHead(nn.Module):
    def __init__(self, hidden_dim, proj_dim,
                 dropout=model_cfg.ATTENTION_DROPOUT):
        super().__init__()
        self.W = nn.Linear(hidden_dim, proj_dim)           # project to subspace
        self.V = nn.Linear(proj_dim, 1, bias=False)        # score
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_output, mask):
        proj = self.W(lstm_output)                         # [B, L, proj_dim]
        energy = torch.tanh(proj)
        scores = self.V(energy).squeeze(-1)                # [B, L]
        scores = scores.masked_fill(
            ~mask, torch.finfo(scores.dtype).min
        )
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Pool from the projected features (not original lstm_output)
        masked_proj = proj * mask.unsqueeze(-1)
        context = torch.bmm(weights.unsqueeze(1), masked_proj).squeeze(1)   # [B, proj_dim]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=model_cfg.NUM_HEADS):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.heads = nn.ModuleList([
            AttentionHead(hidden_dim, head_dim) for _ in range(num_heads)
        ])
        self.out_linear = nn.Linear(head_dim*num_heads, hidden_dim)

    def forward(self, hidden_state, mask):
        x = torch.cat([h(hidden_state, mask) for h in self.heads], dim=-1)
        x = self.out_linear(x)
        return x


class QuoraSiameseClassifier(nn.Module):
    def __init__(self, vocab_size, config=model_cfg, embedding=None):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            vocab_size, config.EMB_DIM, padding_idx=token_cfg.PAD_IDX
        )
        self.emb_norm = nn.LayerNorm(config.EMB_DIM)
        self.emb_dropout = nn.Dropout(config.EMB_DP)

        if embedding is not None:
            if tuple(embedding.shape) != tuple(self.embedding.weight.shape):
                raise ValueError("GloVe embedding shape does not match the vocabulary")
            print("Copying pretrained GloVe into embedding layer...")
            with torch.no_grad():
                self.embedding.weight.copy_(embedding)
            self.embedding.weight.requires_grad = not config.FREEZE_TOKEN_EMBEDDING

        self.LSTM = nn.LSTM(
            input_size=config.EMB_DIM,
            hidden_size=config.HIDDEN_DIM,
            bidirectional=config.BIDIRECTIONAL,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0.0,
            batch_first=True
        )
        self.lstm_norm = nn.LayerNorm(config.LSTM_OUT)
        self.cross_attention = MultiHeadCrossAttention(config.LSTM_OUT)
        self.cross_norm = nn.LayerNorm(config.LSTM_OUT)
        self.attention = MultiHeadAttention(config.LSTM_OUT)  # baseline additive pooling
        self.proj = (
            nn.Linear(config.LSTM_OUT, config.PROJECT_DIM)
            if config.ATTENTION_PROJECTION else nn.Identity()
        )
        self.attn_norm = nn.LayerNorm(config.LSTM_OUT)
        self.fc_dims = self._build_fc_layers(config.INPUT_FC_DIM, config.FC_DIMS, config.FC_DP)

    def _build_fc_layers(self, input_dim, fc_dims, dropout):
        layers = []
        for dim in fc_dims:
            layers.extend((nn.Linear(input_dim, dim), nn.GELU(), nn.Dropout(dropout)))
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)


    def _encode(self, question):
        pad_mask = question.ne(token_cfg.PAD_IDX)
    
        emb = self.embedding(question)
    
        if self.config.LAYER_NORM_EMB:
            emb = self.emb_norm(emb)
    
        emb = self.emb_dropout(emb)
    
        # Process only real tokens in the BiLSTM.
        lengths = pad_mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
    
        encoded, _ = self.LSTM(packed)
    
        out, _ = pad_packed_sequence(
            encoded, batch_first=True,
            total_length=question.size(1)
        )
    
        if self.config.LAYER_NORM_LSTM:
            out = self.lstm_norm(out)
    
        return out, pad_mask

    def _pool(self, hidden, mask):
        pooled = self.attention(hidden, mask)
        if self.config.LAYER_NORM_ATTENTION:
            pooled = self.attn_norm(pooled)
        return self.proj(pooled)

    def forward(self, q1, q2):
        h1, mask1 = self._encode(q1)
        h2, mask2 = self._encode(q2)
        # Both directions in the same forward pass: Q1 <- Q2 and Q2 <- Q1.
        cross1 = self.cross_attention(h1, h2, mask1, mask2)
        cross2 = self.cross_attention(h2, h1, mask2, mask1)
        if self.config.SKIP_CONNECTION:
            h1 = self.cross_norm(h1 + cross1)
            h2 = self.cross_norm(h2 + cross2)
        else:
            h1, h2 = cross1, cross2
        v1 = self._pool(h1, mask1)
        v2 = self._pool(h2, mask2)
        cosine = F.cosine_similarity(v1, v2).unsqueeze(-1)
        features = torch.cat(
            [v1, v2, (v1 - v2).abs(), v1 * v2, cosine], dim=-1
        )
        return self.fc_dims(features).squeeze(-1)



