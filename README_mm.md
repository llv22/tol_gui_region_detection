# Preparation of dino v1 and train the dataset

## Training for models on mobile

Reference:

* [Customize Datasets](https://github.com/llv22/mmdetection_forward/blob/develop/docs/en/advanced_guides/customize_dataset.md)
* [data['category_id'] = self.cat_ids[label] IndexError: list index out of range #4243](https://github.com/open-mmlab/mmdetection/issues/4243)
* [Dataset customization](https://github.com/open-mmlab/mmdetection/tree/master/docs/en)
* [CONFIG](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module)
* [Prepare dataset](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#prepare-datasets)
* [Finetune model](https://mmdetection.readthedocs.io/en/latest/user_guides/finetune.html)

### Setup

1. [configs/dino/convert_mobile_segement_to_coco.py](configs/dino/convert_mobile_segement_to_coco.py) migrates mobile section to coco dataset
generate configs/dino/convert_mobile_segement_to_coco.py and prepare data in configs/dino/data

2. [configs/dino/dino-4scale_r50_8xb2-12e_mobile.py](configs/dino/dino-4scale_r50_8xb2-12e_mobile.py) build model

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-4scale_r50_8xb2-12e_mobile.py
```

### Result on different settings

Reference:

* [Train Object Detector with MMDetection and W&B](https://colab.research.google.com/drive/1-qxf3uuXPJr0QUsIic_4cRLxQ1ZBK3yQ?usp=sharing)
* [Logging analysis](https://mmdetection.readthedocs.io/en/latest/user_guides/useful_tools.html)

extrat components:

```bash
pip install future tensorboard
pip install wandb
```

1. max_epochs = 2, lr=0.0001, in /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/configs/_base_/datasets/mobile_detection.py::train_dataloader & val_dataloader batch_size = 6 

GPU : 42G on A100

Settings: 

```bash
05/20 23:48:53 - mmengine - INFO - Epoch(val) [2][37/37]    coco/bbox_mAP: 0.1760  coco/bbox_mAP_50: 0.2950  coco/bbox_mAP_75: 0.1830  coco/bbox_mAP_s: 0.0000  coco/bbox_mAP_m: 0.1910  coco/bbox_mAP_l: 0.1910  data_time: 0.0209  time: 0.2734

wandb: Run summary:
wandb:          base_lr 0.0001
wandb:    coco/bbox_mAP 0.176
wandb: coco/bbox_mAP_50 0.295
wandb: coco/bbox_mAP_75 0.183
wandb:  coco/bbox_mAP_l 0.191
wandb:  coco/bbox_mAP_m 0.191
wandb:  coco/bbox_mAP_s 0.0
wandb:  d0.dn_loss_bbox 1.07381
wandb:   d0.dn_loss_cls 0.06779
wandb:   d0.dn_loss_iou 0.65904
wandb:     d0.loss_bbox 0.23007
wandb:      d0.loss_cls 0.2753
wandb:      d0.loss_iou 0.2276
wandb:  d1.dn_loss_bbox 1.02076
wandb:   d1.dn_loss_cls 0.05875
wandb:   d1.dn_loss_iou 0.60692
wandb:     d1.loss_bbox 0.22023
wandb:      d1.loss_cls 0.25504
wandb:      d1.loss_iou 0.21373
wandb:  d2.dn_loss_bbox 1.01217
wandb:   d2.dn_loss_cls 0.05644
wandb:   d2.dn_loss_iou 0.59336
wandb:     d2.loss_bbox 0.2115
wandb:      d2.loss_cls 0.24779
wandb:      d2.loss_iou 0.20584
wandb:  d3.dn_loss_bbox 1.00777
wandb:   d3.dn_loss_cls 0.05557
wandb:   d3.dn_loss_iou 0.58736
wandb:     d3.loss_bbox 0.21026
wandb:      d3.loss_cls 0.2383
wandb:      d3.loss_iou 0.20438
wandb:  d4.dn_loss_bbox 1.00768
wandb:   d4.dn_loss_cls 0.05479
wandb:   d4.dn_loss_iou 0.585
wandb:     d4.loss_bbox 0.20878
wandb:      d4.loss_cls 0.2369
wandb:      d4.loss_iou 0.20262
wandb:        data_time 0.02092
wandb:     dn_loss_bbox 1.00779
wandb:      dn_loss_cls 0.0549
wandb:      dn_loss_iou 0.58495
wandb:    enc_loss_bbox 0.27616
wandb:     enc_loss_cls 0.28493
wandb:     enc_loss_iou 0.28675
wandb:            epoch 2
wandb:        grad_norm 23.16829
wandb:             iter 627
wandb:             loss 14.97905
wandb:        loss_bbox 0.20919
wandb:         loss_cls 0.23637
wandb:         loss_iou 0.20246
wandb:               lr 0.0001
wandb:           memory 30647
wandb:             time 0.27344
```


2. max_epochs = 12