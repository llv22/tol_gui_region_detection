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
python tools/train.py configs/dino/dino-4scale_r50_8xb2-12e_mobile.py /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/data/train2017
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

3. max_epochs = 20, lr = 0.001, val_batch_size, train_batch_size = 12, by dino-4scale_r50_8xb2 12e checkpoints

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

After correction of data:

Result: 

```bash
938  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.739
939  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.870
940  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.817
941  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
942  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.689
943  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.755
944  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.834
945  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.869
946  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.869
947  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
948  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.787
949  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.877
950 05/21 07:24:19 - mmengine - INFO - bbox_mAP_copypaste: 0.739 0.870 0.817 0.000 0.689 0.755
951 05/21 07:24:19 - mmengine - INFO - Epoch(val) [20][19/19]    coco/bbox_mAP: 0.7390  coco/bbox_mAP_50: 0.8700  coco/bbox_mAP_75: 0.8170  coco/bbox_mAP_s: 0.0000  coco/bbox_mAP_m: 0.6890  coco/bbox_mAP_l: 0.7550  data_time: 0.0565  time: 0.4842
```

```bash
python tools/test.py configs/dino/dino-4scale_r50_8xb2-12e_mobile.py /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-12e_mobile/epoch_20.pth --show-dir dino-4scale_r50_8xb2-12e_mobile_imgs/
```

result in /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-12e_mobile/20240521_153517/dino-4scale_r50_8xb2-12e_mobile_imgs

4, max_epochs = 20, lr = 0.001, val_batch_size = 3, train_batch_size = 2, loaded from https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth

GPU : 75G on A100

Command:

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-5scale_swin-l_8xb2-36e_mobile.py --train_batch_size 3 --val_batch_size 2 --lr 0.001 --epoch 20 # --train_batch_size 4 out of memory
```

visualize result, refer to [VISUALIZATION](https://mmdetection.readthedocs.io/en/latest/user_guides/visualization.html)

```bash
python tools/test.py configs/dino/dino-5scale_swin-l_8xb2-36e_mobile.py /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-12e_mobile/epoch_20.pth --show-dir dino-4scale_r50_8xb2-12e_mobile_imgs/
```

Result:

```bash
wandb: Run summary:
wandb:          base_lr 0.001
wandb:    coco/bbox_mAP 0.004
wandb: coco/bbox_mAP_50 0.031
wandb: coco/bbox_mAP_75 0.001
wandb:  coco/bbox_mAP_l 0.009
wandb:  coco/bbox_mAP_m 0.0
wandb:  coco/bbox_mAP_s 0.0
wandb:  d0.dn_loss_bbox 1.08605
wandb:   d0.dn_loss_cls 0.14335
wandb:   d0.dn_loss_iou 1.15801
wandb:     d0.loss_bbox 1.01825
wandb:      d0.loss_cls 0.40777
wandb:      d0.loss_iou 1.37544
wandb:  d1.dn_loss_bbox 1.0866
wandb:   d1.dn_loss_cls 0.1724
wandb:   d1.dn_loss_iou 1.15929
wandb:     d1.loss_bbox 0.84916
wandb:      d1.loss_cls 0.42898
wandb:      d1.loss_iou 1.2731
wandb:  d2.dn_loss_bbox 1.05772
wandb:   d2.dn_loss_cls 0.25497
wandb:   d2.dn_loss_iou 1.14394
wandb:     d2.loss_bbox 0.77269
wandb:      d2.loss_cls 0.43709
wandb:      d2.loss_iou 1.20997
wandb:  d3.dn_loss_bbox 1.05863
wandb:   d3.dn_loss_cls 0.26209
wandb:   d3.dn_loss_iou 1.14434
wandb:     d3.loss_bbox 0.77459
wandb:      d3.loss_cls 0.43016
wandb:      d3.loss_iou 1.20601
wandb:  d4.dn_loss_bbox 1.05884
wandb:   d4.dn_loss_cls 0.27255
wandb:   d4.dn_loss_iou 1.1471
wandb:     d4.loss_bbox 0.76086
wandb:      d4.loss_cls 0.44398
wandb:      d4.loss_iou 1.20075
wandb:        data_time 0.00988
wandb:     dn_loss_bbox 1.06063
wandb:      dn_loss_cls 0.25748
wandb:      dn_loss_iou 1.14914
wandb:    enc_loss_bbox 11.3105
wandb:     enc_loss_cls 0.42329
wandb:     enc_loss_iou 2.55191
wandb:            epoch 3
wandb:        grad_norm 83.80324
wandb:             iter 1282
wandb:             loss 43.94187
wandb:        loss_bbox 0.76494
wandb:         loss_cls 0.43017
wandb:         loss_iou 1.19912
wandb:               lr 0.001
wandb:           memory 65078
wandb:             time 3.61632
```

5, max_epochs = 20, lr = 0.001, val_batch_size = 3, train_batch_size = 2, loaded from https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth

GPU : 75G on A100

Command:

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-5scale_swin-l_8xb2-12e_mobile.py --train_batch_size 3 --val_batch_size 2 --lr 0.001 --epoch 20 # --train_batch_size 4 out of memory
```

visualize result, refer to [VISUALIZATION](https://mmdetection.readthedocs.io/en/latest/user_guides/visualization.html)

```bash
python tools/test.py configs/dino/dino-5scale_swin-l_8xb2-12e_mobile /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-12e_mobile/epoch_20.pth --show-dir dino-5scale_swin-l_8xb2-12e_mobile_imgs/
```

Result:

```bash
1983  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.061
1984  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.121
1985  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.058
1986  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
1987  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.024
1988  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.071
1989  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.164
1990  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.252
1991  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.252
1992  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
1993  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.041
1994  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.273
1995 05/22 03:31:51 - mmengine - INFO - bbox_mAP_copypaste: 0.061 0.121 0.058 0.000 0.024 0.071
1996 05/22 03:31:51 - mmengine - INFO - Epoch(val) [20][109/109]    coco/bbox_mAP: 0.0610  coco/bbox_mAP_50: 0.1210  coco/bbox_mAP_75: 0.0580  coco/bbox_mAP_s: 0.0000  coco/bbox_mAP_m: 0.0240  coco/bbox_mAP_l: 0.0710  data_time: 0.0052  time: 0.4794
```

6, max_epochs = 36, lr = 0.001, val_batch_size = 10, train_batch_size = 2, loaded from https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pt

Status: **current best-performed model**

GPU : 75G on A100

Command:

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-4scale_r50_8xb2-36e_mobile.py --train_batch_size 10 --val_batch_size 10 --lr 0.001 --epoch 36 # 12 out of memory during 16
```

Result:

```bash
1406  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.856
1407  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.920
1408  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.886
1409  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
1410  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.846
1411  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.864
1412  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.898
1413  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.915
1414  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.915
1415  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
1416  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.892
1417  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.917
1418 05/22 09:32:29 - mmengine - INFO - bbox_mAP_copypaste: 0.856 0.920 0.886 0.000 0.846 0.864
1419 05/22 09:32:29 - mmengine - INFO - Epoch(val) [36][22/22]    coco/bbox_mAP: 0.8560  coco/bbox_mAP_50: 0.9200  coco/bbox_mAP_75: 0.8860  coco/bbox_mAP_s: 0.0000  coco/bbox_mAP_m: 0.8460  coco/bbox_mAP_l: 0.8640  data_time: 0.0343  time: 0.4361
```

visualize result

```bash
python tools/test.py configs/dino/dino-4scale_r50_8xb2-36e_mobile.py /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-36e_mobile/epoch_36.pth --show-dir dino-4scale_r50_8xb2-36e_mobile_imgs/
```

val result:

```bash
DONE (t=27.55s).
Accumulating evaluation results...
DONE (t=1.26s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.857
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.920
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.886
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.846
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.864
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.898
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.915
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.915
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.892
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.917
05/22 15:44:09 - mmengine - INFO - bbox_mAP_copypaste: 0.857 0.920 0.886 0.000 0.846 0.864
05/22 15:44:09 - mmengine - INFO - Epoch(test) [37/37]    coco/bbox_mAP: 0.8570  coco/bbox_mAP_50: 0.9200  coco/bbox_mAP_75: 0.8860  coco/bbox_mAP_s: 0.0000  coco/bbox_mAP_m: 0.8460  coco/bbox_mAP_l: 0.8640  data_time: 4.0555  time: 4.3898
```

7, large_bbox, max_epochs = 36, lr = 0.001, val_batch_size = 10, train_batch_size = 2, loaded from https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pt

GPU : 75G on A100

Command:

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py configs/dino/dino-4scale_r50_8xb2-36e_mobile_large_bbox.py --train_batch_size 10 --val_batch_size 10 --lr 0.001 --epoch 36 # 12 out of memory during 16
```

Result:

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/test.py configs/dino/dino-4scale_r50_8xb2-36e_mobile_large_bbox.py /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-36e_mobile_large_bbox/epoch_36.pth --show-dir dino-4scale_r50_8xb2-36e_mobile_large_bbox_imgs/
```

Visualization:

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.491
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.777
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.806
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.806                                                                                                                                                                                                                                        ucsc-research-new" 18:48 28-May-24
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.806
05/28 18:49:51 - mmengine - INFO - bbox_mAP_copypaste: 0.413 0.491 0.419 -1.000 -1.000 0.414
05/28 18:49:51 - mmengine - INFO - Epoch(test) [37/37]    coco/bbox_mAP: 0.4130  coco/bbox_mAP_50: 0.4910  coco/bbox_mAP_75: 0.4190  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.4140  data_time: 3.5285  time: 3.8595
```

8, small_bbox, max_epochs = 36, lr = 0.001, val_batch_size = 10, train_batch_size = 2, loaded from https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pt

GPU : 75G on A100

Command:

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-4scale_r50_8xb2-36e_mobile_small_bbox.py --train_batch_size 10 --val_batch_size 10 --lr 0.001 --epoch 36 # 12 out of memory during 16
```

Result:

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/test.py configs/dino/dino-4scale_r50_8xb2-36e_mobile_small_bbox.py /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-36e_mobile_small_bbox/epoch_36.pth --show-dir dino-4scale_r50_8xb2-36e_mobile_small_bbox_imgs/
```

Visualization:

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.720
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.853
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.779
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.751
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.712
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.725
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.825
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.835
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.835
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.871
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.783
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.844
05/28 18:51:54 - mmengine - INFO - bbox_mAP_copypaste: 0.720 0.853 0.779 0.751 0.712 0.725
05/28 18:51:54 - mmengine - INFO - Epoch(test) [37/37]    coco/bbox_mAP: 0.7200  coco/bbox_mAP_50: 0.8530  coco/bbox_mAP_75: 0.7790  coco/bbox_mAP_s: 0.7510  coco/bbox_mAP_m: 0.7120  coco/bbox_mAP_l: 0.7250  data_time: 3.8087  time: 4.1471        
```

9. multi_bbox, max_epochs = 36, lr = 0.001, val_batch_size = 10, train_batch_size = 2, loaded from https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pt

Preparation: 

1, generate new annotations + classes in multiclass label + configuration file
2, change model's head class

GPU : 2*75G on A100

Command:

```bash
python tools/train.py configs/dino/dino-4scale_r50_8xb2-36e_mobile_multi_bbox.py --train_batch_size 10 --val_batch_size 10 --lr 0.001 --epoch 36 # 12 out of memory during 16
# distributed training
export CUDA_VISIBLE_DEVICES=0,1
./tools/dist_train_custom.sh configs/dino/dino-4scale_r50_8xb2-36e_mobile_multi_bbox.py 2
```

Result:

```bash
1328  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.920
1329  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.953
1330  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.929
1331  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.900
1332  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.866
1333  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.921
1334  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.945
1335  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.947
1336  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.947
1337  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.900
1338  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.893
1339  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.948
1340 05/29 00:23:12 - mmengine - INFO - bbox_mAP_copypaste: 0.920 0.953 0.929 0.900 0.866 0.921
1341 05/29 00:23:12 - mmengine - INFO - Epoch(val) [36][11/11]    coco/bbox_mAP: 0.9200  coco/bbox_mAP_50: 0.9530  coco/bbox_mAP_75: 0.9290  coco/bbox_mAP_s: 0.9000  coco/bbox_mAP_m: 0.8660  coco/bbox_mAP_l: 0.9210  data_time: 0.0500  time: 0.4395
```

```bash
python tools/test.py configs/dino/dino-4scale_r50_8xb2-36e_mobile_multi_bbox.py /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-36e_mobile_multi_bbox/epoch_36.pth --show-dir dino-4scale_r50_8xb2-36e_mobile_multi_bbox_imgs/
```

Visualization & Val result:

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.920
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.953
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.929
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.900
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.866
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.921
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.944
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.946
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.946
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.900
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.893
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.948
05/29 07:52:42 - mmengine - INFO - bbox_mAP_copypaste: 0.920 0.953 0.929 0.900 0.866 0.921
05/29 07:52:42 - mmengine - INFO - Epoch(test) [37/37]    coco/bbox_mAP: 0.9200  coco/bbox_mAP_50: 0.9530  coco/bbox_mAP_75: 0.9290  coco/bbox_mAP_s: 0.9000  coco/bbox_mAP_m: 0.8660  coco/bbox_mAP_l: 0.9210  data_time: 3.7632  time: 4.0880
```

10, large_bbox, max_epochs = 12, lr = 0.001, val_batch_size = 2, train_batch_size = 2, loaded from https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth

GPU : 75G on A100

Command:

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/dino/dino-5scale_swin-l_8xb2-12e_mobile_large_bbox.py --train_batch_size 3 --val_batch_size 2 --lr 0.001 --epoch 12 # 12 out of memory during 16
# distributed training
./tools/dist_train_custom.sh configs/dino/dino-5scale_swin-l_8xb2-12e_mobile_large_bbox.py 4
```

Result:

```bash
2266  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.189
2267  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.266
2268  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.171
2269  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
2270  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.003
2271  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.192
2272  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
2273  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.373
2274  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.373
2275  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
2276  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.048
2277  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.379
2278 05/29 05:18:49 - mmengine - INFO - bbox_mAP_copypaste: 0.189 0.266 0.171 0.000 0.003 0.192
2279 05/29 05:18:49 - mmengine - INFO - Epoch(val) [36][28/28]    coco/bbox_mAP: 0.1890  coco/bbox_mAP_50: 0.2660  coco/bbox_mAP_75: 0.1710  coco/bbox_mAP_s: 0.0000  coco/bbox_mAP_m: 0.0030  coco/bbox_mAP_l: 0.1920  data_time: 0.0052  time: 0.5684
```

### Inference on test data

Test data folder: [test_screendata/data](test_screendata/data)

Reference:

* [Inferencer](https://github.com/open-mmlab/mmdetection/blob/main/demo/inference_demo.ipynb)

```bash
python inference_test_screendata.py
```

#### Inference for small_bbox and large_bbox

1. small

```bash
export CUDA_VISIBLE_DEVICES=0
python inference_test_screendata.py --input_folder ../../test_screendata/osworld --model_config configs/dino/dino-4scale_r50_8xb2-36e_mobile_small_bbox.py --checkpoint /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-36e_mobile_small_bbox/epoch_36.pth && python inference_test_screendata.py --input_folder ../../test_screendata/mobile_pc_web --model_config configs/dino/dino-4scale_r50_8xb2-36e_mobile_small_bbox.py --checkpoint /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-36e_mobile_small_bbox/epoch_36.pth
```

2. larger

```bash
export CUDA_VISIBLE_DEVICES=1
python inference_test_screendata.py --input_folder ../../test_screendata/osworld --model_config configs/dino/dino-4scale_r50_8xb2-36e_mobile_large_bbox.py --checkpoint /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-36e_mobile_large_bbox/epoch_36.pth && python inference_test_screendata.py --input_folder ../../test_screendata/mobile_pc_web --model_config configs/dino/dino-4scale_r50_8xb2-36e_mobile_large_bbox.py --checkpoint /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-36e_mobile_large_bbox/epoch_36.pth
```

#### Inference for multi_bbox

```bash
export CUDA_VISIBLE_DEVICES=0
python inference_test_screendata.py --input_folder ../../test_screendata/mobile_pc_web_osworld --model_config configs/dino/dino-4scale_r50_8xb2-36e_mobile_multi_bbox.py --checkpoint /home/xiandao_airs/workspace/ScreenReaderData/models/mmdetection_forward/work_dirs/dino-4scale_r50_8xb2-36e_mobile_multi_bbox/epoch_36.pth
```