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

Command:

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-4scale_r50_8xb2-12e_mobile.py/home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/data/train2017
/home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/data/val2017
```

Result: 

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


2. max_epochs = 2, lr = 0.001, val_batch_size, train_batch_size = 2

GPU : 75G on A100

Command:

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-4scale_r50_8xb2-12e_mobile.py --train_batch_size 12 --val_batch_size 12 --lr 0.001 --epoch 2
```

Result: 

```bash
wandb: Run summary:
wandb:          base_lr 0.001
wandb:    coco/bbox_mAP 0.124
wandb: coco/bbox_mAP_50 0.251
wandb: coco/bbox_mAP_75 0.119
wandb:  coco/bbox_mAP_l 0.139
wandb:  coco/bbox_mAP_m 0.11
wandb:  coco/bbox_mAP_s 0.0
wandb:  d0.dn_loss_bbox 1.20174
wandb:   d0.dn_loss_cls 0.07639
wandb:   d0.dn_loss_iou 0.7493
wandb:     d0.loss_bbox 0.28932
wandb:      d0.loss_cls 0.31339
wandb:      d0.loss_iou 0.28693
wandb:  d1.dn_loss_bbox 1.15717
wandb:   d1.dn_loss_cls 0.07267
wandb:   d1.dn_loss_iou 0.69532
wandb:     d1.loss_bbox 0.27118
wandb:      d1.loss_cls 0.30637
wandb:      d1.loss_iou 0.2647
wandb:  d2.dn_loss_bbox 1.1489
wandb:   d2.dn_loss_cls 0.07023
wandb:   d2.dn_loss_iou 0.67804
wandb:     d2.loss_bbox 0.26803
wandb:      d2.loss_cls 0.29499
wandb:      d2.loss_iou 0.25781
wandb:  d3.dn_loss_bbox 1.14537
wandb:   d3.dn_loss_cls 0.0696
wandb:   d3.dn_loss_iou 0.67024
wandb:     d3.loss_bbox 0.2658
wandb:      d3.loss_cls 0.28702                                                                                                                                                                                                                                                                                                                                                                                                                                      "ucsc-research-new" 00:38 21-May-24
wandb:      d3.loss_iou 0.25485
wandb:  d4.dn_loss_bbox 1.14499
wandb:   d4.dn_loss_cls 0.06978
wandb:   d4.dn_loss_iou 0.66854
wandb:     d4.loss_bbox 0.26572
wandb:      d4.loss_cls 0.28618
wandb:      d4.loss_iou 0.25461
wandb:        data_time 0.04619
wandb:     dn_loss_bbox 1.14504
wandb:      dn_loss_cls 0.06964
wandb:      dn_loss_iou 0.66863
wandb:    enc_loss_bbox 0.37075
wandb:     enc_loss_cls 0.31783
wandb:     enc_loss_iou 0.37811
wandb:            epoch 2
wandb:        grad_norm 16.54684
wandb:             iter 314
wandb:             loss 17.54069
wandb:        loss_bbox 0.26599
wandb:         loss_cls 0.2848
wandb:         loss_iou 0.25472
wandb:               lr 0.001
wandb:           memory 61235
wandb:             time 0.46916
```

3. max_epochs = 20, lr = 0.001, val_batch_size, train_batch_size = 12

GPU : 75G on A100

Command:

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-4scale_r50_8xb2-12e_mobile.py --train_batch_size 12 --val_batch_size 12 --lr 0.001 --epoch 20
```

visualize result, refer to [VISUALIZATION](https://mmdetection.readthedocs.io/en/latest/user_guides/visualization.html)

```bash
python tools/test.py configs/dino/dino-4scale_r50_8xb2-12e_mobile.py /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-12e_mobile/epoch_20.pth --show-dir dino-4scale_r50_8xb2-12e_mobile_imgs/
```

4, max_epochs = 20, lr = 0.001, val_batch_size, train_batch_size = 12

GPU : 75G on A100

Command:

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-5scale_swin-l_8xb2-36e_mobile.py --train_batch_size 4 --val_batch_size 2 --lr 0.001 --epoch 36
```

visualize result, refer to [VISUALIZATION](https://mmdetection.readthedocs.io/en/latest/user_guides/visualization.html)

```bash
python tools/test.py configs/dino/dino-5scale_swin-l_8xb2-36e_mobile.py /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-12e_mobile/epoch_20.pth --show-dir dino-4scale_r50_8xb2-12e_mobile_imgs/
```