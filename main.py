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
#sys.stdout = open(os.devnull, 'w')
#import engine.LangSAM as LangSAM
#import engine.data_process as data_process
import engine.train_ADS as trian_ADS
"""
#import engine.MT as MT
import engine.MT_resize as MT
import engine.OneFormer as OneFormer
import os
#from huggingface_hub import logi# 找img到最接近 start_row 的点n
#login()
import torch
import json
import engine.test_ADS as test_ADS
"""
from imagecorruptions import get_corruption_names


def Equality_MRs(args):
    with open(os.path.join(args.data_file, "MT_results", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    MR = random.choice(MRs)
    args.pre_series = 1
    #corruption_names=['defocus_blur','motion_blur','zoom_blur','snow','gaussian_noise', 'shot_noise', 'impulse_noise',
    # 'frost','fog','brightness','cont                                                                                                                                                                                                                                                                  rast','elastic_transform','pixelate','jpeg_compression']
    corruption_names=['defocus_blur','gaussian_noise', 'fog','brightness','jpeg_compression']
    coorps = ["DeepTest_" + effect for effect in corruption_names]
    #random.shuffle(coorps)
    cont = 0
    data_lists = []
    for i in tqdm(range(len(coorps))):
        MR["idx_new"] ="DeepTest_" + str(i)
        MR["type"] = "Pix2Pix"
        MR["diffusion_prompt"] = coorps[i]
        MR["MR"] = corruption_names[i]
        MR["maneuver"] = "no change"
        #MT.MT(args, MR)
        cont+=1
        data_str = MT.MT_ADS(args, MR)
        print(data_str)
        data_lists.append(data_str)
    test_ADS.save_data_lists_to_json(data_lists,"result.json")

def Test(args,type,pre_series=1):
    args.pre_series = pre_series
    with open(os.path.join("engine",  "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    type_specific_MRs = [mr for mr in MRs if mr["type"] == type]
    for i in range(len(type_specific_MRs)):
        selected_MR = type_specific_MRs[i]# random.choice(type_specific_MRs)
        MT.MT(args, selected_MR)
def Test_1(args,type,pre_series=1):
    args.pre_series = pre_series
    with open(os.path.join("engine",  "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    MT.MT_1(args,type,MRs)


def Test_ADS(args,type="Equality_MRs"):
    with open(os.path.join(args.data_file, "MT_results", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    data_lists = []

    type_specific_MRs = [mr for mr in MRs if mr["type"] == type]
    for i in tqdm(range(5)):
        selected_MR = type_specific_MRs[i]  # random.choice(type_specific_MRs)
        data_str = MT.MT_ADS(args, selected_MR)
        data_lists.append(data_str)
        print(data_str)
    test_ADS.save_data_lists_to_json(data_lists,"result.json")

if __name__ == "__main__":
    args = Paramenters.parse_args()
    #args.dataset = "A2D2"
    args.dataset = "udacity"
    args.Use_time_series=1
    args.data_process = True
    ##########################################
    #data_process.data_process(args) #下采样图像并且配对传感器数据
    #OneFormer.Check_OneFormer(args) #生成道路分割
    #data_process.resize_images(args,"ORA")#把图像变为 320 160 方便训练
    #data_process.prepare_data(args)# 加载为torch结构
    trian_ADS.Train(args)#训练自动驾驶


    #LangSAM.Check_langsam(args)
    #OneFormer.Check_OneFormer_resize(args)  # 生成道路分割
    #############################################
    #Test_datasets = "A2D2"#["A2D2", "udacity"]
    #MT.find_test_images(args)
    #args.dataset =random.choice(Test_datasets)
    #MT.find_test_images(args)

    #Equality_MRs(args)
    #
    #Test_1(args,"diffusion",25) #add_object,add_object_1,Pix2Pix,diffusion
    #Test_1(args, "Pix2Pix", 1)
    #Test_ADS(args, type="Pix2Pix")
   # Test_ADS(args, type="diffusion")

