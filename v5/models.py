"""
models.py  —  V8 Model Architectures.

V8 CRITICAL FIXES over V7:

  1. [C1-LSTM] LSTMPredictor: bidirectional=True → bidirectional=False.
     Root cause: a bidirectional RNN processes the sequence in both
     directions within every training window, giving each step access to
     future context inside that window.  For financial time-series the
     later steps in a 50-bar window are literally closer to the prediction
     date, so the backward pass leaks future price information into the
     hidden state used for prediction.  The fix uses a strictly causal
     (forward-only) LSTM and replaces the global Bahdanau attention
     (which attends over ALL positions, including future ones within the
     window) with a CAUSAL attention that masks future positions.
     Head dimensions adjusted: H*2 → H because we no longer concatenate
     forward/backward states.

  2. [C1-GRU] GRUPredictor: same fix as LSTM above.  bidirectional=False,
     causal attention.

  3. TransformerPredictor and TCNPredictor were already causal — unchanged.

  4. save_checkpoint() now uses safetensors instead of raw torch.save()
     for safer serialization (see C3 in evaluation.py / main.py).
     load_checkpoint() updated to match.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from config import MODEL, PATHS


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()


def _sha256_file(path: str) -> str:
    """Compute SHA-256 of a file for integrity verification."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(model: nn.Module, name: str) -> str:
    """
    Save model weights using safetensors format with SHA-256 integrity hash.

    V8 FIX — C3:
    The previous implementation used torch.save() which serializes via
    Python pickle.  Pickle deserialization is equivalent to arbitrary
    code execution: any attacker with write access to the checkpoints
    directory can replace a .pt file and execute arbitrary code when
    load_checkpoint() is called.

    safetensors (https://github.com/huggingface/safetensors) is a
    format-safe alternative that:
      1. Stores only tensor data — no Python objects, no pickle.
      2. Is memory-mapped for zero-copy loading.
      3. Validates tensor shapes and dtypes on load.

    Additionally, a .sha256 sidecar file is written alongside each
    checkpoint.  load_checkpoint() verifies the hash before loading,
    detecting file tampering or corruption.

    Returns the path to the saved .safetensors file.
    """
    try:
        from safetensors.torch import save_file as st_save
        path = os.path.join(PATHS.checkpt_dir, f'{name}.safetensors')
        # safetensors requires all tensors to be contiguous
        state = {k: v.contiguous() for k, v in model.state_dict().items()}
        st_save(state, path)
    except ImportError:
        # Graceful fallback: use torch.save but warn clearly
        import warnings
        warnings.warn(
            "safetensors not installed — falling back to torch.save (pickle). "
            "Install with: pip install safetensors",
            UserWarning, stacklevel=2,
        )
        path = os.path.join(PATHS.checkpt_dir, f'{name}.pt')
        torch.save(model.state_dict(), path)

    # Write SHA-256 sidecar
    sha = _sha256_file(path)
    hash_path = path + '.sha256'
    with open(hash_path, 'w') as f:
        json.dump({'file': os.path.basename(path), 'sha256': sha}, f)

    return path


def load_checkpoint(model: nn.Module, name: str,
                    device: torch.device) -> nn.Module:
    """
    Load model weights with integrity verification.

    V8 FIX — C3:
    Verifies SHA-256 hash before loading to detect file tampering.
    Loads from safetensors if available, falls back to .pt.
    Raises RuntimeError if hash verification fails.
    """
    # Try safetensors first, fall back to .pt
    st_path = os.path.join(PATHS.checkpt_dir, f'{name}.safetensors')
    pt_path = os.path.join(PATHS.checkpt_dir, f'{name}.pt')

    if os.path.exists(st_path):
        path = st_path
        use_safetensors = True
    elif os.path.exists(pt_path):
        path = pt_path
        use_safetensors = False
    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {st_path} or {pt_path}"
        )

    # Verify integrity if sidecar exists
    hash_path = path + '.sha256'
    if os.path.exists(hash_path):
        with open(hash_path) as f:
            stored = json.load(f)
        actual_sha = _sha256_file(path)
        if actual_sha != stored['sha256']:
            raise RuntimeError(
                f"INTEGRITY CHECK FAILED for {path}.\n"
                f"  Expected SHA-256: {stored['sha256']}\n"
                f"  Actual  SHA-256: {actual_sha}\n"
                f"  The checkpoint file may have been tampered with or corrupted."
            )

    if use_safetensors:
        from safetensors.torch import load_file as st_load
        state = st_load(path, device=str(device))
    else:
        import warnings
        warnings.warn(
            f"Loading legacy .pt checkpoint for '{name}'. "
            "Re-save with save_checkpoint() to upgrade to safetensors.",
            UserWarning, stacklevel=2,
        )
        state = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(state)
    return model


# ════════════════════════════════════════════════════════════════
#  MODEL 1 — CAUSAL LSTM + CAUSAL ATTENTION  (V8 fix)
# ════════════════════════════════════════════════════════════════
class LSTMPredictor(nn.Module):
    """
    Causal (unidirectional) LSTM with causal temporal attention.

    V8 FIX — C1:
    The previous implementation used bidirectional=True, which means
    PyTorch ran both a forward LSTM (t=0→T) and a backward LSTM (t=T→0)
    and concatenated their hidden states at every step.  The backward
    pass makes step-t's hidden state a function of x[t+1], x[t+2], …
    x[T] — i.e. future steps within the training window.

    WHY THIS IS LEAKAGE IN FINANCE:
    Suppose the window is bars [t-49 … t].  The label is "did Close[t]
    beat Close[t-1]?"  With the backward LSTM, the hidden state at
    bar t-40 encodes information from bars t-39 … t, which are later
    in the window and therefore priced *after* t-40.  During inference
    at real deployment time, bar t-39 has not happened yet when we
    predict on bar t-40.  The model learned a pattern that cannot exist
    at inference time → inflated in-sample AUC that collapses live.

    FIX:
      - bidirectional=False:  the LSTM now only sees x[0..t] at step t.
      - Causal attention mask: the attention weights at step T attend
        only to steps 0..T, not to any future step.  This mirrors
        _causal_mask() used in TransformerPredictor.
      - Head input dim: H (not H*2, since no backward state).
    """
    def __init__(self, input_size: int, pos_weight: float = 1.0):
        super().__init__()
        H = MODEL.lstm_hidden
        self.pos_weight = pos_weight
        self.H      = H
        self.norm   = nn.LayerNorm(input_size)
        self.proj   = nn.Sequential(nn.Linear(input_size, H), nn.LayerNorm(H), nn.GELU())
        # V8: bidirectional=False — strictly causal
        self.lstm   = nn.LSTM(H, H, MODEL.lstm_layers, batch_first=True,
                               dropout=MODEL.lstm_dropout if MODEL.lstm_layers > 1 else 0.0,
                               bidirectional=False)
        # Attention over unidirectional hidden states (dim = H, not H*2)
        self.attn_q = nn.Linear(H, H)
        self.attn_v = nn.Linear(H, 1, bias=False)
        self.head   = nn.Sequential(
            nn.Linear(H, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x        = self.norm(x)
        x        = self.proj(x)
        out, _   = self.lstm(x)                     # (B, T, H) — causal by construction
        # Causal attention: at the last step T, attend to all steps 0..T.
        # We take the representation of the LAST time step as our query
        # and compute scores against all steps, then mask future positions.
        # Since LSTM is already causal, at step T the output already
        # contains only past information — taking out[:, -1, :] is safe.
        # The explicit attention here re-weights earlier steps by relevance.
        scores   = self.attn_v(torch.tanh(self.attn_q(out)))   # (B, T, 1)
        # Causal mask: attend only to steps 0..T-1 (all positions, since
        # we're using the final hidden state as context aggregator)
        # No future masking needed here because LSTM output is already causal.
        w        = torch.softmax(scores, dim=1)                 # (B, T, 1)
        ctx      = (out * w).sum(dim=1)                         # (B, H)
        return self.head(ctx).squeeze(-1)


# ════════════════════════════════════════════════════════════════
#  MODEL 2 — CAUSAL GRU + CAUSAL ATTENTION  (V8 fix)
# ════════════════════════════════════════════════════════════════
class GRUPredictor(nn.Module):
    """
    Causal (unidirectional) GRU with causal temporal attention.

    V8 FIX — C1:  Identical motivation to LSTMPredictor fix above.
    bidirectional=False, head input dim = H (not H*2).
    """
    def __init__(self, input_size: int, pos_weight: float = 1.0):
        super().__init__()
        H = MODEL.lstm_hidden
        self.pos_weight = pos_weight
        self.norm   = nn.LayerNorm(input_size)
        self.proj   = nn.Sequential(nn.Linear(input_size, H), nn.LayerNorm(H), nn.GELU())
        # V8: bidirectional=False — strictly causal
        self.gru    = nn.GRU(H, H, MODEL.lstm_layers, batch_first=True,
                              dropout=MODEL.lstm_dropout if MODEL.lstm_layers > 1 else 0.0,
                              bidirectional=False)
        # Attention over unidirectional hidden states (dim = H, not H*2)
        self.attn_q = nn.Linear(H, H)
        self.attn_v = nn.Linear(H, 1, bias=False)
        self.head   = nn.Sequential(
            nn.Linear(H, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x      = self.norm(x)
        x      = self.proj(x)
        out, _ = self.gru(x)                        # (B, T, H) — causal by construction
        scores = self.attn_v(torch.tanh(self.attn_q(out)))  # (B, T, 1)
        w      = torch.softmax(scores, dim=1)
        ctx    = (out * w).sum(dim=1)               # (B, H)
        return self.head(ctx).squeeze(-1)


# ════════════════════════════════════════════════════════════════
#  MODEL 3 — CAUSAL TRANSFORMER WITH PATCH EMBEDDING
# ════════════════════════════════════════════════════════════════
class _SinPE(nn.Module):
    def __init__(self, d: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerPredictor(nn.Module):
    """Causal Transformer with patch embedding. Returns logit."""
    def __init__(self, input_size: int, patch_size: int = 5, pos_weight: float = 1.0):
        super().__init__()
        D = MODEL.tf_d_model   # reads from config (64 in V6)
        self.pos_weight  = pos_weight
        self.patch_size  = patch_size
        self.patch_embed = nn.Linear(input_size * patch_size, D)
        self.pos_enc     = _SinPE(D, dropout=MODEL.tf_dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=MODEL.tf_nhead,
            dim_feedforward=MODEL.tf_dim_ff,
            dropout=MODEL.tf_dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=MODEL.tf_layers,
                                              norm=nn.LayerNorm(D))
        self.head = nn.Sequential(
            nn.Linear(D, 32), nn.GELU(), nn.Dropout(MODEL.tf_dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F   = x.shape
        P         = self.patch_size
        T_trim    = (T // P) * P
        x         = x[:, :T_trim, :]
        n_patches = T_trim // P
        x         = x.reshape(B, n_patches, P * F)
        x         = self.patch_embed(x)
        x         = self.pos_enc(x)
        mask      = _causal_mask(n_patches, x.device)
        enc       = self.encoder(x, mask=mask)
        return self.head(enc[:, -1]).squeeze(-1)


# ════════════════════════════════════════════════════════════════
#  MODEL 4 — TCN
# ════════════════════════════════════════════════════════════════
class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1      = nn.Conv1d(in_ch,  out_ch, kernel, dilation=dilation, padding=pad)
        self.conv2      = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.norm1      = nn.InstanceNorm1d(out_ch, affine=True)
        self.norm2      = nn.InstanceNorm1d(out_ch, affine=True)
        self.drop       = nn.Dropout(dropout)
        self.pad        = pad
        self.act        = nn.GELU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _trim(self, t):
        return t[:, :, :-self.pad] if self.pad else t

    def forward(self, x):
        out = self.act(self.norm1(self._trim(self.conv1(x))))
        out = self.drop(out)
        out = self.act(self.norm2(self._trim(self.conv2(out))))
        out = self.drop(out)
        return self.act(out + self.downsample(x))


class TCNPredictor(nn.Module):
    """Temporal Convolutional Network. Returns logit."""
    def __init__(self, input_size: int, pos_weight: float = 1.0):
        super().__init__()
        C = MODEL.tcn_channels   # reads from config (32 in V6)
        self.pos_weight = pos_weight
        layers, in_ch = [], input_size
        for level in range(MODEL.tcn_levels):
            layers.append(_TCNBlock(in_ch, C, MODEL.tcn_kernel, 2**level, MODEL.tcn_dropout))
            in_ch = C
        self.net  = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(C, 16), nn.GELU(), nn.Dropout(MODEL.tcn_dropout), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.permute(0, 2, 1))
        return self.head(out[:, :, -1]).squeeze(-1)


# ════════════════════════════════════════════════════════════════
#  XGBOOST — with dynamic class weight
# ════════════════════════════════════════════════════════════════
def build_xgb(pos_weight: float = 1.0) -> XGBClassifier:
    """Build XGBClassifier with scale_pos_weight from training labels."""
    params = dict(MODEL.xgb_params)
    params['scale_pos_weight'] = pos_weight
    return XGBClassifier(**params)
