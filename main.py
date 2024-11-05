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

