import os

os.getcwd()
import engine.Paramenters as Paramenters
import engine.OneFormer as OneFormer
import random
import engine.data_process as data_process
import engine.MT as MT
import json
import sys
from tqdm import tqdm
import engine.test_ADS as test_ADS
# sys.stdout = open(os.devnull, 'w')
# import engine.LangSAM as LangSAM
import engine.data_process as data_process
import engine.train_ADS as trian_ADS
from imagecorruptions import get_corruption_names
import torch
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import gc
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from tqdm import tqdm
import numpy as np
import cv2
import imageio
from imagecorruptions import corrupt


def Generate_violation(args, save_path):
    with open(os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    Pix_MRs = [mr for mr in MRs if mr["type"] == "Pix2Pix"]
    diffusion_MRs = [mr for mr in MRs if mr["type"] == "diffusion"]
    length = len(Pix_MRs) + len(diffusion_MRs)
    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)
    test_save_path = os.path.join(save_dir_2, save_path)
    with open(os.path.join(args.data_file, "results", "info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    args.pre_series = 25
    data_dir = os.path.join(args.data_file, "results")
    gap = 10

    for i in range(length):
        break
        save_path_ = os.path.join(save_dir_2, save_path, str(i), "videos")
        test_results = test_ADS.EXP3_test_ads(args, data_dir, save_path_)
        with open(os.path.join(save_dir_2, save_path, str(i), "violation.json"), 'w') as f:
            json.dump(test_results, f)
    save_path_ = os.path.join(args.data_file, "results", "original")
    test_results = test_ADS.EXP3_test_ads(args, data_dir, save_path_)
    with open(os.path.join(args.data_file, "results", "follow_up", "Exp_2", "violation.json"), 'w') as f:
        json.dump(test_results, f)


#########################################################
# Define the gap between images (in pixels)

def A(args, save_path):
    with open(os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    Pix_MRs = [mr for mr in MRs if mr["type"] == "Pix2Pix"]
    diffusion_MRs = [mr for mr in MRs if mr["type"] == "diffusion"]
    length = len(Pix_MRs) + len(diffusion_MRs)
    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)
    test_save_path = os.path.join(save_dir_2, save_path)
    with open(os.path.join(args.data_file, "results", "info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    args.pre_series = 25
    data_dir = os.path.join(args.data_file, "results")
    gap = 10

    with open(os.path.join(args.data_file, "results", "follow_up", "Exp_2", "violation.json"), 'r') as f:
        source_results = json.load(f)

    for i in range(length):
        with open(os.path.join(save_dir_2, save_path, str(i), "violation.json"), 'r') as f:
            follow_results = json.load(f)
        data_process.Check_file(os.path.join(save_dir_2, save_path, str(i), "Ngifs", ))
        save_steering = os.path.join(save_dir_2, save_path, str(i), "steering")
        save_speed = os.path.join(save_dir_2, save_path, str(i), "speed")
        data_process.Check_file(save_steering)
        data_process.Check_file(save_speed)
        cont = 0
        for idx in tqdm(range(0, len(data_lists), args.pre_series)):
            video_dir = os.path.join(save_dir_2, save_path, str(i), "videos")
            pridict_source = source_results[cont*20]
            pridict_follow = follow_results[cont*20]
            frames = []
            for idxx in range(idx, idx + 5):
                original_frame_path = os.path.join(save_dir, os.path.basename(data_lists[idxx]['Image File']))
                generated_frame_path = os.path.join(video_dir, os.path.basename(data_lists[idxx]['Image File']))
                original_frame = cv2.imread(original_frame_path)
                generated_frame = cv2.imread(generated_frame_path)
                original_frame_resized = cv2.resize(original_frame, (320, 160), interpolation=cv2.INTER_AREA)
                generated_frame_resized = cv2.resize(generated_frame, (320, 160), interpolation=cv2.INTER_AREA)

                # 创建文字区域
                text_height = 30
                text_area_original = np.full((text_height, 320, 3), 255, dtype=np.uint8)
                text_area_generated = np.full((text_height, 320, 3), 255, dtype=np.uint8)
                steering_angle_original = pridict_source["steering"]
                steering_angle_generated = pridict_follow["steering"]
                if steering_angle_original > 0:
                    text_original = f"turn left {abs(steering_angle_original):.2f} degrees"
                else:
                    text_original = f"turn right {abs(steering_angle_original):.2f} degrees"

                if steering_angle_generated > 0:
                    text_generated = f"turn left {abs(steering_angle_generated):.2f} degrees"
                else:
                    text_generated = f"turn right {abs(steering_angle_generated):.2f} degrees"
                # 判断是否违法并设置颜色
                if steering_angle_original != 0:
                    change = (steering_angle_generated - steering_angle_original) / steering_angle_original
                else:
                    change = steering_angle_generated

                text_color = (0, 255, 0) if (-0.1 <= change <= 0.1) else (0, 0, 255)
                # 添加文字到区域
                cv2.putText(text_area_original, text_original, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(text_area_generated, text_generated, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                original_with_text = np.vstack((original_frame_resized, text_area_original))
                generated_with_text = np.vstack((generated_frame_resized, text_area_generated))

                # 创建垂直间隔
                total_height = 160 + text_height  # 图片高度 + 文字区域高度
                gap_vertical = np.full((total_height, 10, 3), 255, dtype=np.uint8)

                # 合并成最终帧
                full_frame = np.hstack((original_with_text, gap_vertical, generated_with_text))

                # 转换颜色空间并添加到帧列表
                full_frame_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
                frames.append(full_frame_rgb)
            # Save as GIF
            gif_path = os.path.join(save_dir_2, save_path, str(i), "Ngifs",  f"{cont}.gif")
            imageio.mimsave(gif_path, frames, duration=100, loop=0)
            cont += 1

def B(args, save_path):
    with open(os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    Pix_MRs = [mr for mr in MRs if mr["type"] == "Pix2Pix"]
    diffusion_MRs = [mr for mr in MRs if mr["type"] == "diffusion"]
    length = len(Pix_MRs) + len(diffusion_MRs)
    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)
    test_save_path = os.path.join(save_dir_2, save_path)
    with open(os.path.join(args.data_file, "results", "info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    args.pre_series = 25
    data_dir = os.path.join(args.data_file, "results")
    gap = 10

    with open(os.path.join(args.data_file, "results", "follow_up", "Exp_2", "violation.json"), 'r') as f:
        source_results = json.load(f)

    for i in range(length):
        with open(os.path.join(save_dir_2, save_path, str(i), "violation.json"), 'r') as f:
            follow_results = json.load(f)
        data_process.Check_file(os.path.join(save_dir_2, save_path, str(i), "Ngifs1", ))
        save_steering = os.path.join(save_dir_2, save_path, str(i), "steering")
        save_speed = os.path.join(save_dir_2, save_path, str(i), "speed")
        data_process.Check_file(save_steering)
        data_process.Check_file(save_speed)
        cont = 0
        for idx in tqdm(range(0, len(data_lists), args.pre_series)):
            video_dir = os.path.join(save_dir_2, save_path, str(i), "videos")
            pridict_source = source_results[cont*20]
            pridict_follow = follow_results[cont*20]
            frames = []
            for idxx in range(idx, idx + 5):
                original_frame_path = os.path.join(save_dir, os.path.basename(data_lists[idxx]['Image File']))
                generated_frame_path = os.path.join(video_dir, os.path.basename(data_lists[idxx]['Image File']))
                original_frame = cv2.imread(original_frame_path)
                generated_frame = cv2.imread(generated_frame_path)
                original_frame_resized = cv2.resize(original_frame, (320, 160), interpolation=cv2.INTER_AREA)
                generated_frame_resized = cv2.resize(generated_frame, (320, 160), interpolation=cv2.INTER_AREA)

                # 创建文字区域
                text_height = 30
                text_area_original = np.full((text_height, 320, 3), 255, dtype=np.uint8)
                text_area_generated = np.full((text_height, 320, 3), 255, dtype=np.uint8)
                # 将转向角相关变量改为速度
                speed_original = pridict_source["speed"]
                speed_generated = pridict_follow["speed"]

                # 修改文本显示
                text_original = f"speed {speed_original:.2f} m/s"
                text_generated = f"speed {speed_generated:.2f} m/s"

                # 修改判断逻辑
                if speed_original != 0:
                    change = (speed_generated - speed_original) / speed_original
                else:
                    change = speed_generated

                # 检查是否在允许的减速范围内(-0.3 到 -0.05)
                text_color = (0, 255, 0) if (-0.3 <= change <= -0.05) else (0, 0, 255)

                # 添加文字到区域
                cv2.putText(text_area_original, text_original, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(text_area_generated, text_generated, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                original_with_text = np.vstack((original_frame_resized, text_area_original))
                generated_with_text = np.vstack((generated_frame_resized, text_area_generated))

                # 创建垂直间隔
                total_height = 160 + text_height  # 图片高度 + 文字区域高度
                gap_vertical = np.full((total_height, 10, 3), 255, dtype=np.uint8)

                # 合并成最终帧
                full_frame = np.hstack((original_with_text, gap_vertical, generated_with_text))

                # 转换颜色空间并添加到帧列表
                full_frame_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
                frames.append(full_frame_rgb)
            # Save as GIF
            gif_path = os.path.join(save_dir_2, save_path, str(i), "Ngifs1",  f"{cont}.gif")
            imageio.mimsave(gif_path, frames, duration=100, loop=0)
            cont += 1



# 3906 test results

if __name__ == "__main__":
    args = Paramenters.parse_args()
    save_path = "Exp_2"
    #Generate_violation(args, save_path)
    #A(args, save_path)
    B(args, save_path)
