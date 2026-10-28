# Insulator Defect Dataset — 45-image Subset

A small, class-stratified subset of the degraded insulator defect dataset used
in the paper. It is intended for quick inspection and smoke tests only. The
complete dataset (2,240 train / 640 valid images) will be released on Zenodo
upon publication of the paper; until then it is available from the corresponding
author upon reasonable request.

## Contents

| Split | Images | Broken | Flashover damage | No issues | String |
|-------|--------|--------|------------------|-----------|--------|
| train | 36     | 21     | 14               | 36        | 36     |
| valid | 9      | 7      | 2                | 9         | 9      |

Counts are annotation instances, not distinct images (one image may contain
several classes).

- Images: degraded field photographs, downscaled so the longest side is 1280 px
  (JPEG quality 92) to keep this repository light; originals are ~3008 × 2000 px
  and are preserved in the full dataset archive
- Labels: YOLO detection format, one `.txt` per image (`class cx cy w h`, normalised);
  labels are unchanged by downscaling since coordinates are normalised
- Sampling: random with a fixed seed, with a minimum number of instances per
  class in each split

## Use

```python
from ultralytics import YOLO
model = YOLO("yolov8s.yaml")
model.train(data="insulator_subset/data.yaml", epochs=1, imgsz=640)
```

Or point the training script at it:

```bash
DATA_YAML=insulator_subset/data.yaml python Next_Stage_Sprint/train_db4_inneriou_mosaic50.py
```
