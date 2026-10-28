"""
Route 2: Learnable Frequency-Domain Wavelet Filtering (DWT + SE Attention)

Supported wavelets: haar (db1, 2-tap), db4 (Daubechies-4, 8-tap)

db4 upgrades Haar's step-function characteristic with a smooth 8-tap
convolution kernel, preserving continuous insulator defect edges while
still suppressing diagonal high-frequency noise.

Discrete Wavelet Transform decomposes input into 4 subbands:
  LL — low-frequency structure (insulator topology)
  LH — horizontal edges
  HL — vertical edges
  HH — diagonal high-frequency noise (target for suppression)

A lightweight SE channel attention module learns to reweight fused subband
features, forcing the network to suppress noise-dominated channels while
amplifying structure channels — mathematically, before any CNN convolution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Wavelet filter coefficient registry (populated at import time)
# ---------------------------------------------------------------------------
_WAVELET_CACHE: dict = {}

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


def _get_wavelet_filters(name: str):
    """Return (dec_lo, dec_hi) as float32 tensors for a given wavelet name.

    Sources pywt if available; otherwise falls back to hardcoded db4.
    """
    if name in _WAVELET_CACHE:
        return _WAVELET_CACHE[name]

    if _HAS_PYWT:
        w = pywt.Wavelet(name)
        lo = torch.tensor(w.dec_lo, dtype=torch.float32)
        hi = torch.tensor(w.dec_hi, dtype=torch.float32)
    else:
        # Hardcoded db4 orthonormal decomposition filters
        # (Daubechies-4 with 4 vanishing moments = 8 taps)
        if name == "db4":
            lo = torch.tensor([
                -0.010597401785069032, 0.0328830116668852,
                0.030841381835560764, -0.18703481171909309,
                -0.027983769416859854, 0.6308807679298589,
                0.7148465705529157, 0.2303778133088965
            ], dtype=torch.float32)
            hi = torch.tensor([
                -0.2303778133088965, 0.7148465705529157,
                -0.6308807679298589, -0.027983769416859854,
                0.18703481171909309, 0.030841381835560764,
                -0.0328830116668852, -0.010597401785069032
            ], dtype=torch.float32)
        else:
            raise ValueError(
                f"Wavelet '{name}' requires pywt (not installed) or hardcoded fallback (only 'db4')."
            )

    _WAVELET_CACHE[name] = (lo, hi)
    return lo, hi


# ---------------------------------------------------------------------------
# Haar 2D Discrete Wavelet Transform (orthonormal, sqrt2-normalized)
# Kept for backward compatibility; ~3x faster than generic conv path.
# ---------------------------------------------------------------------------
def haar_dwt_2d(x: torch.Tensor):
    """
    Args:
        x: [B, C, H, W]  (H and W must be even)

    Returns:
        LL, LH, HL, HH — each [B, C, H//2, W//2]
    """
    s2 = math.sqrt(2.0)

    # ---- row transform (along width) ----
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    L = (x_even + x_odd) / s2
    H = (x_even - x_odd) / s2

    # ---- column transform (along height) on L ----
    L_even = L[..., 0::2, :]
    L_odd  = L[..., 1::2, :]
    LL = (L_even + L_odd) / s2
    LH = (L_even - L_odd) / s2

    # ---- column transform on H ----
    H_even = H[..., 0::2, :]
    H_odd  = H[..., 1::2, :]
    HL = (H_even + H_odd) / s2
    HH = (H_even - H_odd) / s2

    return LL, LH, HL, HH


# ---------------------------------------------------------------------------
# Generic convolution-based 2D DWT (for db4 and longer wavelet filters)
# ---------------------------------------------------------------------------
def wavelet_dwt_2d(x: torch.Tensor, wavelet: str = "db4"):
    """
    Separable 2D DWT via convolution + critical downsampling (stride 2).

    Uses asymmetric reflect-padding so output is exactly H//2 x W//2
    regardless of filter length.

    Args:
        x: [B, C, H, W]  (H and W must be even)

    Returns:
        LL, LH, HL, HH — each [B, C, H//2, W//2]
    """
    B, C, H, W = x.shape
    dec_lo, dec_hi = _get_wavelet_filters(wavelet)
    dec_lo = dec_lo.to(device=x.device, dtype=x.dtype)
    dec_hi = dec_hi.to(device=x.device, dtype=x.dtype)

    N = len(dec_lo)  # filter length (8 for db4)
    pad_l = (N - 1) // 2  # left/top   (3 for db4)
    pad_r = N // 2         # right/bot  (4 for db4)

    # Treat each channel as independent "batch" element for conv2d
    x_flat = x.reshape(B * C, 1, H, W)

    lo_row = dec_lo.view(1, 1, 1, N)   # [1, 1, 1, N]
    hi_row = dec_hi.view(1, 1, 1, N)
    lo_col = dec_lo.view(1, 1, N, 1)   # [1, 1, N, 1]
    hi_col = dec_hi.view(1, 1, N, 1)

    # ---- row pass (filter + downsample columns by 2) ----
    x_pad = F.pad(x_flat, (pad_l, pad_r, 0, 0), mode="reflect")
    L = F.conv2d(x_pad, lo_row)[:, :, :, ::2]   # [B*C, 1, H, W//2]
    H = F.conv2d(x_pad, hi_row)[:, :, :, ::2]

    # ---- column pass (filter + downsample rows by 2) ----
    L_pad = F.pad(L, (0, 0, pad_l, pad_r), mode="reflect")
    H_pad = F.pad(H, (0, 0, pad_l, pad_r), mode="reflect")

    LL = F.conv2d(L_pad, lo_col)[:, :, ::2, :]  # [B*C, 1, H//2, W//2]
    LH = F.conv2d(L_pad, hi_col)[:, :, ::2, :]
    HL = F.conv2d(H_pad, lo_col)[:, :, ::2, :]
    HH = F.conv2d(H_pad, hi_col)[:, :, ::2, :]

    # Use .shape to derive target spatial dims — avoids dtype issues with // on
    # older PyTorch where shape elements may not unpack as plain ints.
    HH_out, WW_out = LL.shape[2], LL.shape[3]

    return (
        LL.reshape(B, C, HH_out, WW_out),
        LH.reshape(B, C, HH_out, WW_out),
        HL.reshape(B, C, HH_out, WW_out),
        HH.reshape(B, C, HH_out, WW_out),
    )


def db4_dwt_2d(x: torch.Tensor):
    """Convenience wrapper for Daubechies-4 (8-tap) DWT."""
    return wavelet_dwt_2d(x, wavelet="db4")


# ---------------------------------------------------------------------------
# Inverse 2D DWT (synthesis filterbank)
# ---------------------------------------------------------------------------
def inverse_wavelet_dwt_2d(LL, LH, HL, HH, wavelet: str = "db4"):
    """
    Inverse 2D DWT — reconstructs image from subbands via synthesis filterbank.

    Uses critical upsampling (zero-insertion) followed by synthesis filtering.
    For orthogonal wavelets, synthesis filters = time-reversed analysis filters.

    Args:
        LL, LH, HL, HH: each [B, C, H//2, W//2]
        wavelet: wavelet name (must match forward DWT)

    Returns:
        reconstructed: [B, C, H, W]
    """
    B, C, Hh, Wh = LL.shape
    H, W = Hh * 2, Wh * 2

    dec_lo, dec_hi = _get_wavelet_filters(wavelet)
    rec_lo = dec_lo.flip(0).to(device=LL.device, dtype=LL.dtype)
    rec_hi = dec_hi.flip(0).to(device=LL.device, dtype=LL.dtype)

    N = len(rec_lo)
    pad_t = (N - 1) // 2
    pad_b = N // 2
    pad_l = pad_t
    pad_r = pad_b

    rec_lo_col = rec_lo.view(1, 1, N, 1)   # column filter
    rec_hi_col = rec_hi.view(1, 1, N, 1)
    rec_lo_row = rec_lo.view(1, 1, 1, N)   # row filter
    rec_hi_row = rec_hi.view(1, 1, 1, N)

    def _synthesize_one(subband, col_filt, row_filt):
        # subband: [B*C, 1, Hh, Wh]
        x = subband.reshape(B * C, 1, Hh, Wh)

        # 1. Zero-insertion along height (upsample by 2)
        up_h = torch.zeros(B * C, 1, H, Wh, device=x.device, dtype=x.dtype)
        up_h[:, :, ::2, :] = x
        # 2. Filter along height
        up_h_pad = F.pad(up_h, (0, 0, pad_t, pad_b), mode="reflect")
        col_filt_out = F.conv2d(up_h_pad, col_filt)  # [B*C, 1, H, Wh]

        # 3. Zero-insertion along width (upsample by 2)
        up_w = torch.zeros(B * C, 1, H, W, device=x.device, dtype=x.dtype)
        up_w[:, :, :, ::2] = col_filt_out
        # 4. Filter along width
        up_w_pad = F.pad(up_w, (pad_l, pad_r, 0, 0), mode="reflect")
        row_filt_out = F.conv2d(up_w_pad, row_filt)  # [B*C, 1, H, W]

        return row_filt_out

    recon = (
        _synthesize_one(LL, rec_lo_col, rec_lo_row)
        + _synthesize_one(LH, rec_lo_col, rec_hi_row)
        + _synthesize_one(HL, rec_hi_col, rec_lo_row)
        + _synthesize_one(HH, rec_hi_col, rec_hi_row)
    )

    return recon.reshape(B, C, H, W)


def db4_purify(x: torch.Tensor, suppress_hh: bool = True) -> torch.Tensor:
    """
    db4 wavelet denoising: DWT → keep LL (low-freq structure) → upsample.

    LL subband isolates the low-frequency structural content. Upsampling LL
    back to original resolution gives a noise-suppressed image suitable as
    teacher input — the teacher (trained on clean data) handles the slight
    smoothing gracefully.

    Args:
        x: [B, C, H, W] degraded image tensor
        suppress_hh: unused (kept for API compatibility)

    Returns:
        purified: [B, C, H, W] upsampled-LL image
    """
    LL, _LH, _HL, _HH = wavelet_dwt_2d(x, wavelet="db4")
    return F.interpolate(LL, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation Channel Attention
# ---------------------------------------------------------------------------
class SEAttention(nn.Module):
    """Lightweight SE block: global pooling -> fc -> sigmoid -> channel reweight."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


# ---------------------------------------------------------------------------
# DWT Filter Stem — replaces YOLOv8 Layer 0 (the first Conv)
# ---------------------------------------------------------------------------
class DWTFilterStem(nn.Module):
    """
    Frequency-domain preprocessing stem.

    Flow:
      Input [B, 3, H, W]
        -> DWT (haar or db4) -> LL/LH/HL/HH each [B, 3, H/2, W/2]
        -> Concat   -> [B, 12, H/2, W/2]
        -> 3x3 Conv + BN + SiLU -> [B, c2, H/2, W/2]   (learnable subband fusion)
        -> SE Attention          -> [B, c2, H/2, W/2]   (channel reweight)

    c2 should match the output channel count of YOLOv8's original Layer 0
    (e.g. 32 for YOLOv8s at width=0.50).

    wavelet_type: 'haar' (2-tap, fast, sharper edges) or
                  'db4'  (8-tap, smooth, preserves continuous defect edges)
    """

    def __init__(self, c1: int, c2: int, reduction: int = 4, wavelet_type: str = "db4"):
        super().__init__()
        mid_ch = c1 * 4  # 4 subbands x input channels
        self.wavelet_type = wavelet_type

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.se = SEAttention(c2, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Frequency decomposition
        if self.wavelet_type == "haar":
            subbands = haar_dwt_2d(x)
        else:
            subbands = wavelet_dwt_2d(x, wavelet=self.wavelet_type)

        # 2. Concatenate all subbands along channel axis
        stacked = torch.cat(subbands, dim=1)  # [B, 4*c1, H/2, W/2]

        # 3. Learnable subband fusion + activation
        feat = self.fuse(stacked)  # [B, c2, H/2, W/2]

        # 4. Channel attention -> learn to suppress noise-dominant channels
        feat = self.se(feat)

        return feat


def create_dwt_stem(c1: int = 3, c2: int = 32, wavelet_type: str = "db4") -> DWTFilterStem:
    """Factory: create a DWTFilterStem with the requested wavelet."""
    return DWTFilterStem(c1, c2, wavelet_type=wavelet_type)


def make_dwt_trainer(wavelet_type: str = "db4", pretrained_weights: str = None):
    """Return a DetectionTrainer subclass that swaps layer 0 for DWTFilterStem.

    The swap must happen inside the trainer: ultralytics rebuilds the model from
    the YAML cfg in BaseTrainer.setup_model(), so replacing model.model[0] before
    model.train() is silently discarded.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class DWTDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)

            if pretrained_weights:
                ckpt = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
                obj = ckpt.get("ema") or ckpt.get("model")
                if obj is not None:
                    sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
                else:
                    sd = ckpt.get("state_dict", ckpt)
                state = model.state_dict()
                compatible = {k: v for k, v in sd.items()
                              if k in state and state[k].shape == v.shape}
                model.load_state_dict(compatible, strict=False)
                print(f"[WEIGHTS] Loaded {len(compatible):,} / {len(state):,} tensors "
                      f"from {pretrained_weights}")

            old = model.model[0]
            if not isinstance(old, DWTFilterStem):
                stem = DWTFilterStem(old.conv.in_channels, old.conv.out_channels,
                                     wavelet_type=wavelet_type)
                stem.i, stem.f, stem.type = old.i, old.f, "DWTFilterStem"
                model.model[0] = stem
                with torch.no_grad():
                    probe = stem(torch.zeros(1, old.conv.in_channels, 64, 64))
                assert probe.shape[2:] == (32, 32), f"Stem shape mismatch: {probe.shape}"
                print(f"[STEM] DWTFilterStem({wavelet_type}) injected: "
                      f"{sum(p.numel() for p in stem.parameters()):,} params")

            return model

    return DWTDetectionTrainer
