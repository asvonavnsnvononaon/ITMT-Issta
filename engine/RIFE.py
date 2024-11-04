import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前目录（ECCV2022）的父目录（Diffusion_tools）添加到 Python 路径
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir,"Diffusion_tools","ECCV2022"))


from My_image_generation import interpolate_images


def interpolate(img_path1, img_path2, model_dir="../Diffusion_tools/ECCV2022/train_log"):
    fig_new = interpolate_images(img_path1, img_path2, model_dir="Diffusion_tools/ECCV2022/train_log")
    return fig_new
