"""Policy heads for state-only drone action prediction.

Both models accept identical inputs and return the same two heads:
    state: (B, H, STATE_DIM)  float
    mode:  (B, H)             long, phase id
    ->     cont:        (B, T, LABEL_DIM)   continuous actions
           grid_logits: (B, T, GRID_TOTAL)  grid_idx classification

This shared interface lets the same trainer drive either architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from state_policy.dataset import (
    GRID_TOTAL,
    LABEL_DIM,
    PHASES,
    STATE_DIM,
)


N_PHASES = len(PHASES)


def _build_mode_embed(out_dim: int):
    return nn.Embedding(N_PHASES + 1, out_dim)  # +1 catches the unknown id (-1)


def _clean_mode(mode: torch.Tensor) -> torch.Tensor:
    """Replace -1 (unknown) with N_PHASES so the embedding lookup is in range."""
    return mode.clamp_min(0).where(mode >= 0, torch.full_like(mode, N_PHASES))


# --------------------------- MLP baseline ---------------------------

class StateMLP(nn.Module):
    """Flatten (history × (state + mode_embed)) then 2-layer MLP per head."""

    def __init__(self, history_len: int, horizon: int,
                 hidden: int = 256, mode_embed: int = 8, dropout: float = 0.0):
        super().__init__()
        self.history_len = history_len
        self.horizon = horizon
        self.mode_embed = _build_mode_embed(mode_embed)
        in_dim = history_len * (STATE_DIM + mode_embed)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
        )
        self.cont_head = nn.Linear(hidden, horizon * LABEL_DIM)
        self.grid_head = nn.Linear(hidden, horizon * GRID_TOTAL)

    def forward(self, state: torch.Tensor, mode: torch.Tensor):
        B = state.shape[0]
        m = self.mode_embed(_clean_mode(mode))                       # (B, H, d_m)
        x = torch.cat([state, m], dim=-1).reshape(B, -1)
        h = self.trunk(x)
        cont = self.cont_head(h).view(B, self.horizon, LABEL_DIM)
        grid_logits = self.grid_head(h).view(B, self.horizon, GRID_TOTAL)
        return cont, grid_logits


# --------------------------- Transformer ---------------------------

class StateTransformer(nn.Module):
    """Encoder over (history_tokens + horizon_query_tokens) with self-attention.

    Each history step contributes one token (linear projection of state + mode
    embedding + learnable position). Horizon outputs are learnable query
    tokens that attend to history and to each other through the encoder.
    The last `horizon` tokens of the final layer feed two heads.
    """

    def __init__(self, history_len: int, horizon: int,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 ffn_dim: int = 256, dropout: float = 0.1,
                 mode_embed: int = 8):
        super().__init__()
        self.history_len = history_len
        self.horizon = horizon
        self.d_model = d_model

        # Token construction.
        self.state_proj = nn.Linear(STATE_DIM, d_model)
        self.mode_embed = _build_mode_embed(mode_embed)
        self.mode_proj = nn.Linear(mode_embed, d_model)

        self.history_pos = nn.Embedding(history_len, d_model)
        # Two learnable query components: 'identity' tells the layer which
        # future step this token represents; 'position' is its index.
        self.query_embed = nn.Embedding(horizon, d_model)
        self.query_pos = nn.Embedding(horizon, d_model)

        # Pre-norm GELU encoder. Full self-attention over H+T tokens.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

        self.cont_head = nn.Linear(d_model, LABEL_DIM)
        self.grid_head = nn.Linear(d_model, GRID_TOTAL)

    def forward(self, state: torch.Tensor, mode: torch.Tensor):
        B, H, _ = state.shape
        T = self.horizon
        device = state.device

        m = self.mode_proj(self.mode_embed(_clean_mode(mode)))         # (B, H, d)
        s = self.state_proj(state)                                     # (B, H, d)
        h_pos = self.history_pos(torch.arange(H, device=device))       # (H, d)
        hist_tokens = s + m + h_pos.unsqueeze(0)                       # (B, H, d)

        q_idx = torch.arange(T, device=device)
        q_tokens = (self.query_embed(q_idx) + self.query_pos(q_idx))   # (T, d)
        q_tokens = q_tokens.unsqueeze(0).expand(B, -1, -1)             # (B, T, d)

        seq = torch.cat([hist_tokens, q_tokens], dim=1)                # (B, H+T, d)
        out = self.encoder(seq)
        future = self.out_norm(out[:, -T:, :])                         # (B, T, d)

        cont = self.cont_head(future)                                  # (B, T, 5)
        grid_logits = self.grid_head(future)                           # (B, T, 49)
        return cont, grid_logits


def build_model(arch: str, history_len: int, horizon: int, **kwargs) -> nn.Module:
    """Factory: 'mlp' or 'transformer'. Extra kwargs forwarded to the class."""
    arch = arch.lower()
    if arch == "mlp":
        allowed = {"hidden", "mode_embed", "dropout"}
        kw = {k: v for k, v in kwargs.items() if k in allowed}
        return StateMLP(history_len=history_len, horizon=horizon, **kw)
    if arch in ("transformer", "tx"):
        allowed = {"d_model", "n_heads", "n_layers", "ffn_dim", "dropout", "mode_embed"}
        kw = {k: v for k, v in kwargs.items() if k in allowed}
        return StateTransformer(history_len=history_len, horizon=horizon, **kw)
    raise ValueError(f"Unknown arch: {arch!r} (use 'mlp' or 'transformer')")
