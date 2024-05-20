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

1. Epoch 1 

Settings: 

```bash
05/20 22:37:55 - mmengine - INFO - Epoch(train) [2][300/327]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:00:28  time: 1.0907  data_time: 0.0116  memory: 31877  grad_norm: 22.5916  loss: 14.5960  loss_cls: 0.2240  loss_bbox: 0.1985  loss_iou: 0.1939  d0.loss_cls: 0.2697  d0.loss_bbox: 0.2104  d0.loss_iou: 0.2116  d1.loss_cls: 0.2466  d1.loss_bbox: 0.2059  d1.loss_iou: 0.2021  d2.loss_cls: 0.2360  d2.loss_bbox: 0.1999  d2.loss_iou: 0.1965  d3.loss_cls: 0.2289  d3.loss_bbox: 0.1961  d3.loss_iou: 0.1934  d4.loss_cls: 0.2255  d4.loss_bbox: 0.1969  d4.loss_iou: 0.1934  enc_loss_cls: 0.2765  enc_loss_bbox: 0.2607  enc_loss_iou: 0.2732  dn_loss_cls: 0.0551  dn_loss_bbox: 0.9960  dn_loss_iou: 0.5717  d0.dn_loss_cls: 0.0679  d0.dn_loss_bbox: 1.0620  d0.dn_loss_iou: 0.6526  d1.dn_loss_cls: 0.0586  d1.dn_loss_bbox: 1.0119  d1.dn_loss_iou: 0.5961  d2.dn_loss_cls: 0.0571  d2.dn_loss_bbox: 0.9994  d2.dn_loss_iou: 0.5797  d3.dn_loss_cls: 0.0557  d3.dn_loss_bbox: 0.9957  d3.dn_loss_iou: 0.5742  d4.dn_loss_cls: 0.0550  d4.dn_loss_bbox: 0.9957  d4.dn_loss_iou: 0.5718
05/20 22:38:24 - mmengine - INFO - Exp name: dino-4scale_r50_8xb2-12e_mobile_20240520_222537
05/20 22:38:24 - mmengine - INFO - Saving checkpoint at 2 epochs
05/20 22:38:38 - mmengine - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.32s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=35.15s).
Accumulating evaluation results...
DONE (t=1.23s).

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.170
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.283
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.180
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.184
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.190
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.443
05/20 22:39:15 - mmengine - INFO - bbox_mAP_copypaste: 0.170 0.283 0.176 0.000 0.180 0.184
05/20 22:39:15 - mmengine - INFO - Epoch(val) [2][37/37]    coco/bbox_mAP: 0.1700  coco/bbox_mAP_50: 0.2830  coco/bbox_mAP_75: 0.1760  coco/bbox_mAP_s: 0.0000  coco/bbox_mAP_m: 0.1800  coco/bbox_mAP_l: 0.1840  data_time: 0.0164  time: 0.2547
```