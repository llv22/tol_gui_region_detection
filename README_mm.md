# Preparation of dino v1 and train the dataset

## Data format

1. prepare annotation_coco.json from configs/dino

## train the model

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-4scale_r50_8xb2-12e_mobile.py
```