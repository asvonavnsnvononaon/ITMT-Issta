
import engine.data_process as data_process
import cv2
import os
import matplotlib.pyplot as plt
import torch
import tqdm
import torch
from PIL import Image
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录的父目录添加到 Python 路径
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir,"Other_tools","evf-sam2"))
import argparse
import os
import sys

import os
import torch
import cv2
import numpy as np
from transformers import AutoTokenizer
from model.evf_sam2 import EvfSam2Model
from inference import sam_preprocess, beit3_preprocess
from model.evf_sam import EvfSamModel

def load_and_preprocess_image(image_path):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np
@torch.no_grad()
def Check_langsam(args):
    if args.dataset == "udacity":
        dir_1 = os.path.join(args.data_file, "ADS_data", "udacity")
        dir_2 = os.path.join(args.data_file, "ADS_data", "udacity", "langsam","roadside")
        dir_3 = os.path.join(args.data_file, "ADS_data", "udacity", "langsam", "road")
        datasets = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
    else:
        dir_1 = os.path.join(args.data_file, "ADS_data", "A2D2")
        dir_2 = os.path.join(args.data_file, "ADS_data", "A2D2", "langsam","roadside")
        dir_3 = os.path.join(args.data_file, "ADS_data", "A2D2", "langsam", "road")
        datasets = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]
    sam="sam2"
    if sam == "sam2":
        model_path = "YxZhang/evf-sam2"
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", use_fast=False)
        kwargs = {"torch_dtype": torch.half}
        image_model = EvfSam2Model.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        image_model = image_model.eval().to(args.device)
        model_type = "sam2"

    else:
        version = "YxZhang/evf-sam"
        model_type = "ori"

        tokenizer = AutoTokenizer.from_pretrained(
            version,
            padding_side="right",
            use_fast=False,
        )

        kwargs = {
            "torch_dtype": torch.half,
        }
        model = EvfSamModel.from_pretrained(version, low_cpu_mem_usage=True,
                                            **kwargs).eval()
        model.to('cuda')

    for i in range(len(datasets)):
        print(datasets[i])
        file_1 = os.path.join(dir_1, datasets[i], "center")
        file_2 = os.path.join(dir_2, datasets[i], "center")
        file_3 = os.path.join(dir_3, datasets[i], "center")
        data_process.Check_file(file_2)
        data_process.Check_file(file_3)
        if args.dataset == "udacity":
            Source_images = data_process.get_all_image_paths_Udacity(file_1)
        else:
            Source_images = data_process.get_all_image_paths_A2D2(file_1)
        for img_file in tqdm.tqdm(Source_images):
            img_name = os.path.basename(img_file)
            file_1_path = os.path.join(file_1, img_name)
            file_2_path = os.path.join(file_2, img_name)
            file_3_path = os.path.join(file_3, img_name)
            with torch.no_grad():
                image_np = load_and_preprocess_image(file_1_path)
                # Preprocess image
                original_size_list = [image_np.shape[:2]]
                image_beit = beit3_preprocess(image_np, 224).to(dtype=image_model.dtype,
                                                                device=image_model.device)

                image_sam, resize_shape = sam_preprocess(image_np, model_type=model_type)
                image_sam = image_sam.to(dtype=image_model.dtype,
                                         device=image_model.device)

                # Process text prompt
                prompt_1 = "An object on the right side of the image, off the road"
                input_ids = tokenizer(prompt_1, return_tensors="pt")["input_ids"].to(device=image_model.device)
                # Inference
                pred_mask = image_model.inference(
                    image_sam.unsqueeze(0),
                    image_beit.unsqueeze(0),
                    input_ids,
                    resize_list=[resize_shape],
                    original_size_list=original_size_list,
                )
                pred_mask = pred_mask.detach().cpu().numpy()[0]
                binary_mask_road = np.where(pred_mask > 0, 255, 0).astype(np.uint8)
                cv2.imwrite(file_2_path, binary_mask_road)
                # Process text prompt
                prompt_2 = "An object in the middle of the road, not the road"#"One vehicle closest to the center"
                input_ids = tokenizer(prompt_2, return_tensors="pt")["input_ids"].to(device=image_model.device)
                # Inference
                pred_mask = image_model.inference(
                    image_sam.unsqueeze(0),
                    image_beit.unsqueeze(0),
                    input_ids,
                    resize_list=[resize_shape],
                    original_size_list=original_size_list,
                )
                pred_mask = pred_mask.detach().cpu().numpy()[0]
                binary_mask_road = np.where(pred_mask > 0, 255, 0).astype(np.uint8)
                cv2.imwrite(file_3_path, binary_mask_road)
    # Preprocess image

    image_sam, resize_shape = sam_preprocess(image_np, model_type="sam2")
    original_size_list = [image_np.shape[:2]]
    image_beit = beit3_preprocess(image_np, 224).to(dtype=image_model.dtype, device=image_model.device)
    image_sam, resize_shape = sam_preprocess(image_np, model_type="sam2")
    image_sam = image_sam.to(dtype=image_model.dtype, device=image_model.device)


    print(1)


"""
   model = LangSAM()
    for i in range(len(datasets)):
        file_1 = os.path.join(dir_1, datasets[i], "center")
        file_2 = os.path.join(dir_2, datasets[i], "center")
        file_3 = os.path.join(dir_3, datasets[i], "center")
        data_process.Check_file(file_2)
        data_process.Check_file(file_3)
        print(datasets[i])
        if args.dataset == "udacity":
            Source_images = data_process.get_all_image_paths_Udacity(file_1)
        else:
            Source_images = data_process.get_all_image_paths_A2D2(file_1)
        for img_file in tqdm.tqdm(Source_images):
            img_name = os.path.basename(img_file)
            file_1_path = os.path.join(file_1, img_name)
            file_2_path = os.path.join(file_2, img_name)
            file_3_path = os.path.join(file_3, img_name)
            with torch.no_grad():
                image_pil = Image.open(file_1_path).convert("RGB")
                text_prompt = "The object on the right of the image"
                masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
                mask = masks[0, :, :].cpu().numpy()
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(file_2_path)

                text_prompt = "The object closest to the center of the image"
                masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
                mask = masks[0, :, :].numpy()
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(file_3_path)


            #The object closest to the center of the image
"""