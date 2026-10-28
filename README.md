# Insulator Defect Detection under Image Degradation

Reference implementation of **a learnable db4 wavelet stem** and **Inner-CIoU**
for YOLOv8s-based insulator defect detection on degraded images.

The method combines three components:

| Component | Role |
|-----------|------|
| **DWTFilterStem (db4)** | Replaces YOLOv8 layer 0. Decomposes the input into LL/LH/HL/HH sub-bands, fuses them with a 3×3 Conv, and reweights channels with SE attention before any CNN convolution. |
| **Inner-CIoU (ratio = 0.75)** | Replaces CIoU in `BboxLoss`. Computing IoU on a centre-scaled inner box amplifies the regression gradient for small defects while staying inside the CIoU metric space, so `TaskAlignedAssigner` keeps its standard CIoU matching. |
| **Mosaic schedule** | Mosaic is enabled for epochs 0–50 and disabled afterwards, protecting degraded fine detail in the later training stage. |

Default training settings: 200 epochs, `imgsz=768`, AdamW (`lr0=1e-3`, cosine), batch size probed from GPU memory.

## Repository layout

```
Next_Stage_Sprint/
  dwt_filter.py                    DWT stem, SE attention, trainer hook
  inner_iou_loss.py                Inner-CIoU and BboxLoss patch
  train_db4_inneriou_mosaic50.py   training entry point
  yolov8s-wavelet.yaml             network definition (nc=4)
data/
  data.yaml                        full-dataset configuration (relative paths)
insulator_subset/
  data.yaml, train/, valid/        45-image stratified subset with labels
```

## Dataset

| Split | Images | Labels |
|-------|--------|--------|
| train | 2,240  | 2,240  |
| valid | 640    | 640    |

- 4 classes: `0 Broken`, `1 Flashover damage`, `2 No issues`, `3 String`
- Images are degraded field photographs, approximately 3008 × 2000 px
- Labels are standard YOLO detection format (`class cx cy w h`, normalised)

**Dataset availability.** This repository includes a 45-image stratified subset
(`insulator_subset/`) for inspection and quick experiments. The complete dataset
(2,880 images: 2,240 train / 640 valid) will be released on Zenodo upon
publication of the paper. Until then, the full dataset is available from the
corresponding author upon reasonable request.

Place the downloaded `train/` and `valid/` folders under a `dataset/` directory
at the repository root:

```
insulator-defect-detection/
├── dataset/
│   ├── train/images  train/labels
│   └── valid/images  valid/labels
├── data/data.yaml
└── Next_Stage_Sprint/
```

## Environment

- Python 3.10+
- PyTorch 2.x with CUDA (batch size is probed automatically: 16 / 8 / 4)
- `pip install -r requirements.txt`

`PyWavelets` is optional: the db4 decomposition coefficients are hard-coded as a
fallback when `pywt` is unavailable.

## Training

Run from the repository root (the script imports the `Next_Stage_Sprint` package
relative to the working directory):

```bash
python Next_Stage_Sprint/train_db4_inneriou_mosaic50.py
```

Paths can be overridden without editing the script:

```bash
# Linux / macOS
DATA_YAML=/path/to/data.yaml MODEL_YAML=Next_Stage_Sprint/yolov8s-wavelet.yaml \
  python Next_Stage_Sprint/train_db4_inneriou_mosaic50.py

# Windows (cmd)
set DATA_YAML=D:\datasets\insulator\data.yaml
python Next_Stage_Sprint/train_db4_inneriou_mosaic50.py
```

Outputs are written to `runs/detect/Next_Stage_Sprint/yolov8s_db4_inneriou_mosaic50/`.

## Implementation notes

- **Stem injection must happen inside the trainer.** Ultralytics rebuilds the
  model from the YAML in `BaseTrainer.setup_model()`, so replacing `model.model[0]`
  before `model.train()` is silently discarded. `make_dwt_trainer()` subclasses
  `DetectionTrainer` and swaps layer 0 in `get_model()`.
- **Only the loss is patched.** `patch_loss_for_inner_iou()` monkey-patches
  `ultralytics.utils.loss.BboxLoss.forward`; the assigner is untouched.
  `restore_loss()` reverts the patch after training.
- **Augmentation is deliberately mild** (`scale=0.3`, reduced HSV, no mixup) so
  that the synthetic augmentation does not compound the image degradation.

## Citation

```bibtex
@article{insulator_db4_innerciou,
  title   = {...},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

*Update once the manuscript is published.*

## Contact

1497322011@qq.com
