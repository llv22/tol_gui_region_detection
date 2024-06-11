import os
import json
import argparse
from pathlib import Path
from mmdet.apis import DetInferencer
from time import time

def conf():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_folder", type=str, default="../../test_screendata/mobile_pc_web_osworld")
    argparser.add_argument("--model_config", type=str, default="configs/dino/dino-4scale_r50_8xb2-90e_mobile_multi_bbox.py")
    argparser.add_argument("--checkpoint", type=str, default="work_dirs/dino-4scale_r50_8xb2-90e_mobile_multi_bbox/epoch_90.pth")
    argparser.add_argument("--batch_size", type=int, default=64)
    argparser.add_argument("--img_type", type=str, default="png;jpg")
    return argparser.parse_args()

if __name__ == "__main__":
    args = conf()
    conf_file = Path(args.model_config)
    input_file = Path(args.input_folder)
    input_folder_name = input_file.stem
    output_folder_path = Path(f"{input_file.parent.__str__()}/output_{conf_file.stem}_{input_folder_name}")
    if not output_folder_path.exists():
        output_folder_path.mkdir(parents=True)
    output_folder_path_s = output_folder_path.__str__()
    
    if not Path(args.model_config).exists() or not Path(args.checkpoint).exists():
        print("Please provide valid model config and checkpoint paths")
        exit(1)
        
    cuda_id = os.getenv("CUDA_VISIBLE_DEVICES", "-1")
    if cuda_id == "-1":
        device = "cpu"
    else:
        device = "cuda"
    inferencer = DetInferencer(model=args.model_config, weights=args.checkpoint, device=device, show_progress=True)    
    summary_f = f"{output_folder_path_s}/summary.json"
    output_dir = str(output_folder_path)
    input_folder = Path(args.input_folder)
    start = time()
    image_types = args.img_type.split(";")
    with open(summary_f, "w") as f:
        img_full_paths = []
        for img in image_types:
            img_full_paths += [str(img) for img in input_folder.glob(f"*.{img}")]
        img_full_paths = sorted(img_full_paths)
        infer_results = inferencer(img_full_paths, out_dir=output_dir, batch_size=args.batch_size)
        img_names = [Path(img).name for img in img_full_paths]
        results = dict(zip(img_names, infer_results['predictions']))
        for img_name in img_names:
            relative_output_dir = output_dir.replace("../../", "")
            results[img_name]['visualization'] = f"{relative_output_dir}/vis/{img_name}" 
        json.dump(results, f, indent=1)
        print(f"Results are saved in {summary_f} with time taken: {time() - start} seconds.")