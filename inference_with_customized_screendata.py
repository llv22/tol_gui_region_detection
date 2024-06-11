import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

def conf():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_folder", type=str, default="../../test_screendata/mobile_pc_web_osworld")
    argparser.add_argument("--threshold", type=float, default=0.3)
    argparser.add_argument("--annotation_file", type=str, default="../../test_screendata/output_dino-4scale_r50_8xb2-90e_mobile_multi_bbox_mobile_pc_web_osworld/summary.json")
    argparser.add_argument("--output_folder", type=str, default="../../test_screendata/output_dino-4scale_r50_8xb2-90e_mobile_multi_bbox_mobile_pc_web_osworld/customized_vis")
    return argparser.parse_args()

if __name__ == "__main__":
    args = conf()
    annotation_file = Path(args.annotation_file)
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    if not annotation_file.exists() or not input_folder.exists():
        print("Please provide valid annotation file and input folder paths")
        exit(1)
        
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
        
    with open(annotation_file) as f:
        annotation = json.load(f)
        for pic_name, pic_data in annotation.items():
            pic_path = f"{input_folder}/{pic_name}"
            # "labels", "scores", "bboxes"
            val_pic_data = [index for index, v in enumerate(pic_data["scores"]) if v > args.threshold]
            val_pic_data = [{"labels": pic_data["labels"][index], "bboxes": pic_data["bboxes"][index]} for index in val_pic_data]
            img = Image.open(pic_path)
            draw = ImageDraw.Draw(img, "RGBA")
            for data in val_pic_data:
                label, bbox = data["labels"], data["bboxes"]
                if label == 0:
                    draw.rectangle(bbox, outline="green", width=3)
                else:
                    draw.rectangle(bbox, outline="red", width=5)
            output_path = f"{output_folder}/{pic_name}"
            img.save(output_path)