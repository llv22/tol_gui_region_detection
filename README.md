# ToL Hierarchical GUI region detection

Our ToL Hierarchical GUI region detection model is based on [mmdetection](https://github.com/open-mmlab/mmdetection). We have finetuned DINO with a customized configuration on Android Screen Hierarchical Layout (ASHL) dataset and inference on [Screen Point-and-Read (ScreenPR) Benchmark](https://huggingface.co/datasets/yfan1997/ScreenPR). This guide covers how to set up environment, training and inference details.

## 1. Environment setup

You need to prepare mmdetection environment based on our cloned source code.

* Step 1: Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

* Step 2: Install MMDetection from our source repository

```bash
cd <the root of repo tol_gui_region_detection>
pip install -v -e . -r requirements/tracking.txt
```

* Step 3: Install extra components to support sync results on [wandb.io](https://wandb.ai/):

```bash
pip install future tensorboard
pip install wandb
```

## 2. Training ToL model on ASHL dataset

* Step 1 [Optional]: prepare training data with coco style using the migration script [configs/dino/convert_mobile_segement_to_multilabel_coco.py](configs/dino/convert_mobile_segement_to_multilabel_coco.py). Supposed the training data has been put into [../data/screendata](../data/screendata) folder. As we also put the generated files [configs/dino/data/train/annotation_multilabel_coco.json](configs/dino/data/train/annotation_multilabel_coco.json) and [configs/dino/data/val/annotation_multilabel_coco.json](configs/dino/data/val/annotation_multilabel_coco.json) into our source code, this step can be optional if you don't need configuration different from us.

```bash
cd configs/dino/
python convert_mobile_segement_to_multilabel_coco.py
```

* Step 2, Using [./tools/dist_train_custom_multi_bbox.sh](./tools/dist_train_custom_multi_bbox.sh) to train model on multiple GPUs using Rest backbone. The model configuration file is [configs/dino/dino-4scale_r50_8xb2-90e_mobile_multi_bbox.py](configs/dino/dino-4scale_r50_8xb2-90e_mobile_multi_bbox.py). For our cases, 4 * A6000 are used and you can change the dist_train_custom_multi_bbox.sh based on your own machine settings.

Run the following script to train on  4 * A6000:

```bash
# distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3
./tools/dist_train_custom_multi_bbox.sh configs/dino/dino-4scale_r50_8xb2-90e_mobile_multi_bbox.py 4
```

On wandb.ai, the result after 90 epoch as follow:

```bash
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.941
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.962
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.947
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.702
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.897
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.943
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.959
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.961
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.961
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.814
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.916
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.963
mmengine - INFO - bbox_mAP_copypaste: 0.941 0.962 0.947 0.702 0.897 0.943
mmengine - INFO - Epoch(val) [90][11/11]    coco/bbox_mAP: 0.9410  coco/bbox_mAP_50: 0.9620  coco/bbox_mAP_75: 0.9470  coco/bbox_mAP_s: 0.7020  coco/bbox_mAP_m: 0.8970  coco/bbox_mAP_l: 0.9430  data_time: 0.0137  time: 0.2778
```

You can use the following script to run test.py for test data and the visualization result will be saved in the folder [dino-4scale_r50_8xb2-90e_mobile_multi_bbox_imgs/](dino-4scale_r50_8xb2-90e_mobile_multi_bbox_imgs/).

```bash
python tools/test.py configs/dino/dino-4scale_r50_8xb2-90e_mobile_multi_bbox.py ./work_dirs/dino-4scale_r50_8xb2-90e_mobile_multi_bbox/epoch_90.pth --show-dir dino-4scale_r50_8xb2-90e_mobile_multi_bbox_imgs/
```

* Step 3 [Optional]: use Swin-l as backbone to train for 12 epoch with configuration file [configs/dino/dino-5scale_swin-l_8xb2-36e_mobile_multi_bbox.py](configs/dino/dino-5scale_swin-l_8xb2-36e_mobile_multi_bbox.py). In comparison, the loss curve is much worse than the one of Rest backbone. 

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python tools/train.py configs/dino/dino-5scale_swin-l_8xb2-36e_mobile_multi_bbox.py --train_batch_size 2 --val_batch_size 2 --lr 0.001 --epoch 12 # 12 out of memory during 16
# distributed training
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train_custom_multi_bbox.sh configs/dino/dino-5scale_swin-l_8xb2-36e_mobile_multi_bbox.py 4
```

## 3. Inference on ScreenPR dataset

* Step 1: Data preparation

Put ScreenPR dataset under [the src folder](https://github.com/UeFan/Screen-Point-and-Read/tree/main/src) of Screen-Point-and-Read github folder, having the relative path of [../../../data/mobile_pc_web_osworld]([../../mobile_pc_web_osworld](https://github.com/eric-ai-lab/Screen-Point-and-Read/tree/main/data/mobile_pc_web_osworld)) to the root of current github project.

* Step 2: Using our trained ToL model

The pretrained LoT weight has been shared in [DINO weights trained by 90 epoch](https://drive.google.com/file/d/1IN3EfDKyXwu5WegqyFOWfXH6ttJ3zNdx/view?usp=sharing), save it to [./work_dirs/dino-4scale_r50_8xb2-90e_mobile_multi_bbox/epoch_90.pth](./work_dirs/dino-4scale_r50_8xb2-90e_mobile_multi_bbox/epoch_90.pth) and use the following script to trigger inference. A output folder will be generated with the name output_dino-4scale_r50_8xb2-90e_mobile_multi_bbox_mobile_pc_web_osworld under the same parent folder [../../../data/](https://github.com/eric-ai-lab/Screen-Point-and-Read/tree/main/data/mobile_pc_web_osworld).  

```bash
export CUDA_VISIBLE_DEVICES=0
python inference_test_screendata.py --input_folder ../../../data/mobile_pc_web_osworld --model_config configs/dino/dino-4scale_r50_8xb2-90e_mobile_multi_bbox.py --checkpoint ./work_dirs/dino-4scale_r50_8xb2-90e_mobile_multi_bbox/epoch_90.pth
```

* Step 3: Using original Dino model

Download [the original Dino weights](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth) and  save it to [./work_dirs/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth](./work_dirs/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth) and use the following script to trigger inference.

```bash
export CUDA_VISIBLE_DEVICES=0
python inference_test_screendata_by_dino_original.py --input_folder ../../../data/mobile_pc_web_osworld
```

### Using ToL model trained before

## Reference

* [mmdetection preparation](https://mmdetection.readthedocs.io/en/latest/get_started.html)
* [Customize Datasets](https://github.com/llv22/mmdetection_forward/blob/develop/docs/en/advanced_guides/customize_dataset.md)
* [Dataset customization](https://github.com/open-mmlab/mmdetection/tree/master/docs/en)
* [CONFIG](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module)
* [Prepare dataset](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#prepare-datasets)
* [Finetune model](https://mmdetection.readthedocs.io/en/latest/user_guides/finetune.html)
* [Train Object Detector with MMDetection and W&B](https://colab.research.google.com/drive/1-qxf3uuXPJr0QUsIic_4cRLxQ1ZBK3yQ?usp=sharing)
* [Logging analysis](https://mmdetection.readthedocs.io/en/latest/user_guides/useful_tools.html)
* [Inferencer on mmdetection DINO](https://github.com/open-mmlab/mmdetection/blob/main/demo/inference_demo.ipynb)
* [Deal with the issue "data['category_id'] = self.cat_ids[label] IndexError: list index out of range #4243"](https://github.com/open-mmlab/mmdetection/issues/4243)