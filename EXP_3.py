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
#import engine.LangSAM as LangSAM
import engine.data_process as data_process
import engine.train_ADS as trian_ADS
from imagecorruptions import get_corruption_names
import torch
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import gc
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import torch
from torchvision import transforms
from PIL import Image
import os
import engine.sota as sota
import engine.test_ADS as test_ADS

def exp_3_2(args,save_path,type):
    #compare rain When SUT replaces weather into rain, egovehicle should maintain the steering angle.
    with open(os.path.join(args.data_file, "results","info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    #1.our method
    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)
    test_save_path = os.path.join(save_dir_2, save_path, str(type))
    input_dir = save_dir
    mask_dir = save_dir_1
    output_dir = test_save_path
    data_process.Check_file(test_save_path)
    if type ==1:
        diffusion_prompt = "rainy day"
        pix2pix = MT.Pix2PixProcessor(strength=8)
        args.pre_series = 25
        for idx in range(0, len(data_lists), args.pre_series):
            start_index = idx
            end_index = idx + args.pre_series
            MT.EXP_Pix2Pix(pix2pix, data_lists, input_dir, mask_dir, output_dir, start_index, end_index, diffusion_prompt)
    # 2.DeepRoad
    if type==2:
        sota.day2rain(input_dir, output_dir)
    # 3.DeepTest
    if type==3:
        sota.gen_rain(input_dir, output_dir)

def exp_3_1(args,save_path,type):
    #compare rain When SUT replaces weather into rain, egovehicle should maintain the steering angle.
    with open(os.path.join(args.data_file, "results","info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    #1.our method
    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)
    test_save_path = os.path.join(save_dir_2, save_path, str(type))
    input_dir = save_dir
    mask_dir = save_dir_1
    output_dir = test_save_path
    data_process.Check_file(test_save_path)
    if type ==1:
        diffusion_prompt = "You are driving on a road. Ahead of you, some pedestrians is walking on the road"
        inpainter = MT.FluxInpainting(image_size=512)
        args.pre_series = 25
        for idx in range(0, len(data_lists), args.pre_series):
            start_index = idx
            end_index = idx + args.pre_series
            MT.EXP_diffusion(inpainter, data_lists, input_dir, mask_dir, output_dir, start_index, end_index,
                             diffusion_prompt)
        gc.collect()
        torch.cuda.empty_cache()
        sys.path.append(os.path.join("Other_tools", "Vista"))
        from my_test import VideoGenerator
        sys.stdout = open(os.devnull, 'w')

        vista_gen = VideoGenerator()
        for idx in range(0, len(data_lists), args.pre_series):
            start_index = idx
            end_index = idx + args.pre_series
            MT.EXP_vista(args,vista_gen, data_lists, input_dir, mask_dir, output_dir, start_index, end_index, diffusion_prompt)

        # 2.RMT
    #source_image_path
    if type==2:
        source_image_path = os.path.join("SOTA", "Images", "RMT.png")
        sota.add_person(input_dir,mask_dir, output_dir,source_image_path)
    # 3.Metasem
    if type==3:
        source_image_path = os.path.join("SOTA", "Images",  "Metasem.png")
        sota.add_person(input_dir,mask_dir, output_dir,source_image_path)

def Exp_ADS(args,data_dir):
    save_path = os.path.join(data_dir,"original")
    speed_loss = test_ADS.Test(args,data_dir, save_path,pre_models= ["speed"])
    criterion = torch.nn.L1Loss(reduction='sum')
    steering_loss = test_ADS.Test(args,data_dir, save_path,pre_models= ["steering"])
    for i in range(len(speed_loss)):
        print(f"model name: {speed_loss[i]['model']}")
        print(f"Speed loss:{criterion(speed_loss[i]['output'],speed_loss[i]['label'])/len(speed_loss[i]['label'])}")
        print(f"steering loss:{criterion(steering_loss[i]['output'], steering_loss[i]['label']) / len(steering_loss[i]['label'])}")


def resize_png_in_folder(folder_path, size=(320, 160)):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)

            try:
                with Image.open(file_path) as img:
                    # Resize image
                    resized_img = img.resize(size, Image.LANCZOS)

                    # Save resized image, overwriting the original
                    resized_img.save(file_path, 'PNG')
                print(f"Resized {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
def Exp(args, data_dir, Exp_type="Exp_1"):
    input_dir = os.path.join(data_dir, "original")
    if Exp_type=="Exp_1":
        #loss = test_ADS.Test(args, data_dir, input_dir, pre_models=["speed"])
        Models = ["ITMT","RMT","Metasem"]
        for i in range(3):
            output_dir = os.path.join(data_dir, "follow_up", "Exp_3_1", str(i+1))
            results = test_ADS.get_violation(args,data_dir,input_dir,output_dir,maneuver="slow down")
            print(f"model:{Models[i]},results:{results}")
    if Exp_type=="Exp_2":
        #loss = test_ADS.Test(args, data_dir, input_dir, pre_models=["speed"])
        Models = ["ITMT","DeepRoad","DeepTest"]
        for i in range(3):
            output_dir = os.path.join(data_dir, "follow_up", "Exp_3_2", str(i+1))
            results = test_ADS.get_violation(args,data_dir,input_dir,output_dir,maneuver="keep the same")
            print(f"model:{Models[i]},results:{results}")
def exp_3(data_dir):
    for i in range(3):
        output_dir = os.path.join(data_dir, "follow_up", "Exp_3_1", str(i + 1))
        resize_png_in_folder(output_dir)
        output_dir = os.path.join(data_dir, "follow_up", "Exp_3_2", str(i + 1))
        resize_png_in_folder(output_dir)
"""
model:ITMT,results:[{'model': 'Resnet101', 'violations': 4072}, {'model': 'Vgg16', 'violations': 3345}, {'model': 'Epoch', 'violations': 3051}, {'model': 'PilotNet', 'violations': 2849}, {'model': 'CNN_LSTM', 'violations': 2860}, {'model': 'CNN_3D', 'violations': 3053}]
model:RMT,results:[{'model': 'Resnet101', 'violations': 3178}, {'model': 'Vgg16', 'violations': 2589}, {'model': 'Epoch', 'violations': 2479}, {'model': 'PilotNet', 'violations': 747}, {'model': 'CNN_LSTM', 'violations': 1224}, {'model': 'CNN_3D', 'violations': 1693}]
model:Metasem,results:[{'model': 'Resnet101', 'violations': 3287}, {'model': 'Vgg16', 'violations': 2586}, {'model': 'Epoch', 'violations': 2323}, {'model': 'PilotNet', 'violations': 511}, {'model': 'CNN_LSTM', 'violations': 1132}, {'model': 'CNN_3D', 'violations': 1296}]
=1

model:ITMT,results:[{'model': 'Resnet101', 'violations': 4515}, {'model': 'Vgg16', 'violations': 4430}, {'model': 'Epoch', 'violations': 2807}, {'model': 'PilotNet', 'violations': 4449}, {'model': 'CNN_LSTM', 'violations': 4408}, {'model': 'CNN_3D', 'violations': 4321}]

=2
model:ITMT,results:[{'model': 'Resnet101', 'violations': 4545}, {'model': 'Vgg16', 'violations': 4464}, {'model': 'Epoch', 'violations': 2858}, {'model': 'PilotNet', 'violations': 4511}, {'model': 'CNN_LSTM', 'violations': 4435}, {'model': 'CNN_3D', 'violations': 4360}]
=3
model:ITMT,results:[{'model': 'Resnet101', 'violations': 4529}, {'model': 'Vgg16', 'violations': 4481}, {'model': 'Epoch', 'violations': 2909}, {'model': 'PilotNet', 'violations': 4518}, {'model': 'CNN_LSTM', 'violations': 4463}, {'model': 'CNN_3D', 'violations': 4375}]

model:DeepRoad,results:[{'model': 'Resnet101', 'violations': 4507}, {'model': 'Vgg16', 'violations': 4512}, {'model': 'Epoch', 'violations': 2595}, {'model': 'PilotNet', 'violations': 4583}, {'model': 'CNN_LSTM', 'violations': 4503}, {'model': 'CNN_3D', 'violations': 4410}]
model:DeepTest,results:[{'model': 'Resnet101', 'violations': 4058}, {'model': 'Vgg16', 'violations': 3675}, {'model': 'Epoch', 'violations': 2421}, {'model': 'PilotNet', 'violations': 3781}, {'model': 'CNN_LSTM', 'violations': 4090}, {'model': 'CNN_3D', 'violations': 3853}]

"""
if __name__ == "__main__":
    args = Paramenters.parse_args()
    #exp_3_1(args, save_path="Exp_3_1", type=1)
    #exp_3_1(args, save_path="Exp_3_1", type=2)
    exp_3_1(args, save_path="Exp_3_1", type=3)
    #exp_3_2(args, save_path="Exp_3_2", type=1)
    # exp_3_2(args,save_path="Exp_3_2",type=2)
    #exp_3_2(args, save_path="Exp_3_2", type=3)

    ########First 获得所有图片关于方向盘和速度的损失###########
    #data_dir = os.path.join(args.data_file, "results")
    #output_dir = os.path.join(data_dir, "follow_up", "Exp_3_2", str( 1))
    #resize_png_in_folder(output_dir)
   # exp_3(data_dir)
    #Exp_ADS(args, data_dir)
    #Exp(args, data_dir, Exp_type="Exp_1")
    Exp(args,data_dir, Exp_type="Exp_1")

    #type=1


    #save_path = "Exp_3_1"
    #save_path_ = os.path.join(data_dir,"follow_up",save_path, str(type))
    #pre_models = ["speed", "steering"]
    #pre_models = ["speed"]
    #results = test_ADS.Test(args,data_dir, save_path_,pre_models)
    #print(1)