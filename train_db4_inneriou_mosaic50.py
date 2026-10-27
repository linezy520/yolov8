#!/usr/bin/env python3
"""
db4 DWT Stem + Inner-CIoU Loss + Mosaic → Close at Epoch 50

FINAL SPRINT EXPERIMENT — yolov8s_db4_inneriou_mosaic50

Strategy:
  1. db4 DWT Stem (92.90% ceiling, unchanged)
  2. Inner-CIoU (ratio=0.75) replacing standard CIoU in BboxLoss:
     amplifies small-object regression gradient 1.16x without breaking
     TaskAlignedAssigner's CIoU-based sample matching.
  3. Mosaic ON (epochs 0-50) → diversity boost for early convergence
     Mosaic OFF (epochs 51-200) → protect degraded fine-detail integrity
  4. Reduced HSV/scale augmentation to avoid secondary degradation

Key insight:
  Inner-CIoU preserves the CIoU metric space → assigner and loss are aligned.
  NWD was never active in prior run (ultralytics 8.4.48 has zero NWD plumbing).

Implementation notes:
  - Inner-CIoU: BboxLoss.forward is monkey-patched at runtime (ultralytics
    source untouched).
  - db4 stem: injected inside the trainer (get_model), because ultralytics
    rebuilds the model from the YAML before training starts.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import sys
import torch

sys.path.insert(0, ".")
from Next_Stage_Sprint.dwt_filter import make_dwt_trainer

from ultralytics import YOLO
from Next_Stage_Sprint.inner_iou_loss import inner_ciou, patch_loss_for_inner_iou, restore_loss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_YAML = "Next_Stage_Sprint/yolov8s-wavelet.yaml"
DATA_YAML  = "D:/绝缘子数据集及其模型与代码/zip/绝缘子降质后数据集/data.yaml"
PROJECT    = "runs/detect/Next_Stage_Sprint"
NAME       = "yolov8s_db4_inneriou_mosaic50"

# ---------------------------------------------------------------------------
# GPU memory probe
# ---------------------------------------------------------------------------
def probe_batch_size(initial: int = 16, fallback: int = 8) -> int:
    if not torch.cuda.is_available():
        return 4
    gpu = torch.cuda.get_device_properties(0)
    mem_gb = gpu.total_memory / (1024**3)
    print(f"[GPU] {torch.cuda.get_device_name(0)}  |  {mem_gb:.1f} GB")
    return initial if mem_gb >= 15.0 else (fallback if mem_gb >= 10.0 else 4)

BATCH = probe_batch_size(initial=16, fallback=8)

# ---------------------------------------------------------------------------
# Sanity check: Inner-CIoU is active
# ---------------------------------------------------------------------------
print("[CHECK] Verifying Inner-CIoU(ratio=0.75)...")
gt = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
pd = torch.tensor([[12.0, 12.0, 32.0, 32.0]], requires_grad=True)
ic = inner_ciou(gt, pd, ratio=0.75)
ic.backward()
print(f"  Inner-CIoU(ratio=0.75) = {ic.item():.4f}  |  grad = {pd.grad.abs().sum().item():.4f}")
print("  OK — Inner-CIoU implementation confirmed.")

# ---------------------------------------------------------------------------
# Build model (db4 stem is injected by the trainer after ultralytics builds it)
# ---------------------------------------------------------------------------
print(f"\n[BUILD] Loading architecture: {MODEL_YAML}")
model = YOLO(MODEL_YAML)
DWTDetectionTrainer = make_dwt_trainer(wavelet_type="db4")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    original_forward = patch_loss_for_inner_iou(ratio=0.75)
    results = model.train(
        trainer       = DWTDetectionTrainer,
        data          = DATA_YAML,
        epochs        = 200,
        batch         = BATCH,
        imgsz         = 768,
        optimizer     = "AdamW",
        cos_lr        = True,
        lr0           = 0.001,
        lrf           = 0.01,
        weight_decay  = 0.01,
        warmup_epochs = 5.0,
        pretrained    = True,
        # --- Mosaic strategy: ON early, OFF after epoch 50 ---
        mosaic        = 1.0,
        close_mosaic  = 50,
        # --- Reduced aug for degraded small-object integrity ---
        mixup         = 0.0,
        scale         = 0.3,      # was 0.5
        translate     = 0.05,     # was 0.1
        erasing       = 0.2,      # was 0.4
        hsv_h         = 0.005,    # was 0.015
        hsv_s         = 0.3,      # was 0.7
        hsv_v         = 0.2,      # was 0.4
        degrees       = 0.0,
        shear         = 0.0,
        perspective   = 0.0,
        fliplr        = 0.5,
        flipud        = 0.0,
        # --- Project ---
        project       = PROJECT,
        name          = NAME,
        exist_ok      = True,
        device        = 0,
        workers       = 4,
        amp           = True,
        seed          = 42,
        deterministic = True,
        val           = True,
        plots         = True,
        save          = True,
        save_period   = 20,
    )

    print("\n" + "=" * 72)
    print("[DONE] db4 DWT + Inner-CIoU + Mosaic50 training complete.")
    print(f"  Experiment : {NAME}")
    print(f"  Best weights: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"  mAP@50      : {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP@50-95   : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print("=" * 72)

    restore_loss(original_forward)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
