import os.path as osp
import argparse

import mmcv

from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress
from pathlib import Path

def convert_mobile_segment_to_coco(ann_file, out_file, image_prefix):
    data_infos = load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, (k, v) in enumerate(track_iter_progress(list(data_infos.items()))):
        filename = k.replace(".json", ".jpeg")[1:]
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(id=idx, file_name=filename, height=height, width=width))

        bbox = []; categories = []
        for category_id, label in enumerate(["small_bbox", "large_bbox"]):
            categories.append({
                'id': category_id,
                'name': label
            })
            for x, y, w, h, control_id in v[label]:
                if (x, y, w, h) not in bbox:
                    bbox.append((x, y, w, h))
                    poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    poly = [p for x in poly for p in x]
                    data_anno = dict(
                        image_id=idx,
                        id=obj_count,
                        category_id=category_id,
                        bbox=[x, y, w, h],
                        area=h * w,
                        control_id=control_id,
                        json_file=k,
                        # segmentation=[poly],
                        iscrowd=0)
                    annotations.append(data_anno)
                    obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)
    dump(coco_format_json, out_file, indent=1)

def conf():
    args = argparse.ArgumentParser()
    args.add_argument("--train_ann_file", type=str, default="../../../../ScreenReaderData_train.json")
    args.add_argument("--train_out_file", type=str, default="data/train/annotation_multilabel_coco.json")
    args.add_argument("--val_ann_file", type=str, default="../../../../ScreenReaderData_val.json")
    args.add_argument("--val_out_file", type=str, default="data/val/annotation_multilabel_coco.json")
    args.add_argument("--image_prefix", type=str, default="../../../../")
    return args.parse_args()

def refine_file_name(filePath:Path, label:str):
    return f"{filePath.parent}/{filePath.stem}_{label}{filePath.suffix}"

if __name__ == '__main__':
    args = conf()
    train_out_file = args.train_out_file
    val_out_file = args.val_out_file
    convert_parameters = [
        (args.train_ann_file, train_out_file, args.image_prefix),
        (args.val_ann_file, val_out_file, args.image_prefix)
    ]
    for ann_file, out_file, image_prefix in convert_parameters:
        convert_mobile_segment_to_coco(ann_file=ann_file, out_file=out_file, image_prefix=image_prefix)