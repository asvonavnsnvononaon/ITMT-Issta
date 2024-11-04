import os

#from Diffusion_tools.diffusion.Inpainting import image_path

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
import pandas as pd
#start_idx = 0, 1000, 2000
import cv2
import random
import numpy as np
import numpy
from torch.utils.data import Dataset, ConcatDataset, random_split
from torch.utils.data import TensorDataset, DataLoader
from imagecorruptions import corrupt
import transformers
from diffusers import FluxPipeline

sys.path.append(os.path.join( "Other_tools", "Vista"))


from my_test import VideoGenerator
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def cleanup_pytorch_resources():
    # 删除所有还存在的局部变量

    for name in list(locals()):
        del locals()[name]

    # 清空 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 清除没有被引用的循环垃圾
    gc.collect()

    # 重置峰值内存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


set_seed(42)  # 或任何固定的数字
def find_tests(args):

    output_dir = os.path.join(args.data_file, "results", "follow_up", "Exp_4")
    MT.find_test_images_1000(args,output_dir)
class EXP_Class_1:
    def __init__(self,args,Test_type="Exp_1"):
        self.data_dir = os.path.join(args.data_file, "results", "follow_up", "Exp_4")
        self.args = args
        if Test_type == "Exp_1":
            self.data_process = [0, 1, 2, 4, 5, 6, 7,  9, 10, 11]
            self.maneuver = "slow down"
            self.alpha, self.beta = -0.3, 0
            self.args.pre_models = ["speed"]
        else:
            self.data_process = [0, 1, 2, 5,6,7,8,10,11,14]
            self.maneuver = "keep the same"
            self.alpha, self.beta = -0.1, 0.1
            self.args.pre_models = ["steering"]
        self.matched_data = pd.read_excel(os.path.join(self.data_dir, "matched_data.xlsx"), sheet_name="Matched Data")
        self.matched = self.matched_data.iloc[:]
        self.all_test_number = len(self.matched)
        self.Test_type = Test_type
        with open(os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
            self.MRs = json.load(file)
        self.test_round=4
        self.bMR = []
        self.wMR = []

        self.save_place = os.path.join(self.data_dir,Test_type)
        self.Test_type = Test_type

        with open(os.path.join(self.data_dir, "info.json"), 'r', encoding='utf-8') as file:
            self.data_lists = json.load(file)


    def get_violation(self,under_test):
        violation_tests=[]

        for i in range(len(under_test)):
            results = test_ADS.get_violation(args,self.data_dir,os.path.join(self.data_dir, "original"),os.path.join(self.data_dir, "MRs", str(under_test[i])),maneuver=self.maneuver)
            total_violations = sum(item['violations'] for item in results)
            violation_tests.append(total_violations)

        return violation_tests

    def apply_match(self,MR_dir,after,type):
        output_dir =os.path.join(self.data_dir,self.Test_type,f"{os.path.basename(MR_dir)}_{after}")
        data_process.Check_file(output_dir)
        input_dir = MR_dir
        mask_dir =os.path.join(self.data_dir,"mask")
        diffusion_prompt = self.MRs[after]["diffusion_prompt"]

        if self.MRs[after]["type"]=="Pix2Pix":
            pix2pix = MT.Pix2PixProcessor(strength=2)
            for idx in range(0, len(self.data_lists),25):
                start_index = idx
                end_index = idx+25
                #image_path = os.path.join(MR_dir, os.path.basename(row['Image File']))
                MT.EXP_Pix2Pix(pix2pix, self.data_lists , input_dir, mask_dir, output_dir, start_index, end_index,
                               diffusion_prompt)
            cleanup_pytorch_resources()
        elif self.MRs[after]["type"]=="diffusion":
            if type == "best":
                prompt_input = ", ".join(map(str, self.bMR)) + ", " + diffusion_prompt
            if type == "worst":
                prompt_input = ", ".join(map(str, self.wMR)) + ", " + diffusion_prompt
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            file_path = os.path.join(args.data_file, "results", "follow_up", "Exp_2",str(after), "batch.json")
            with open(file_path, 'r') as file:
                data = json.load(file)
            diffusion_prompts = []
            for idx in range(0, len(self.data_lists),25):
                short_prompt = prompt_input
                diffusion_prompt = "low view 1.5," + "You are driving on a road" + short_prompt
                break
                start_index = idx//25

                device = "cpu"
                model_checkpoint = "gokaygokay/Flux-Prompt-Enhance"
                tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
                enhancer = pipeline('text2text-generation',
                                    model=model,
                                    tokenizer=tokenizer,
                                    repetition_penalty=1.2,
                                    device=device)
                max_target_length = 256
                prefix = "enhance prompt: "
                generated_texts = data["generated_text"][idx // 25]
                short_prompt = prompt_input
                diffusion_prompt = "low view 1.5," + "You are driving on a road" +short_prompt
                answer = enhancer(prefix + short_prompt, max_length=max_target_length)
                diffusion_prompt = answer[0]['generated_text']
                diffusion_prompt = "low view 1.5,"+"You are driving on a road"+diffusion_prompt

                diffusion_prompts.append(diffusion_prompt)



            cleanup_pytorch_resources()

            inpainter = MT.FluxInpainting(image_size=256)
            for idx in range(0, len(self.data_lists),25):
                image_path = os.path.join(input_dir,os.path.basename(self.data_lists[idx]['Image File']))
                mask_path = os.path.join(mask_dir, os.path.basename(self.data_lists[idx]['Image File']))
                image = inpainter(image_path, mask_path, prompt=diffusion_prompt)
                image.save(os.path.join(output_dir, os.path.basename(self.data_lists[idx]['Image File'])))

            cleanup_pytorch_resources()
            sys.stdout = open(os.devnull, 'w')
            vista_gen = VideoGenerator()
            for idx in tqdm(range(0, len(self.data_lists), 25)):
                start_index = idx
                image_path = os.path.join(output_dir, os.path.basename(self.data_lists[start_index]['Image File']))
                speed = self.data_lists[start_index]['Vehicle Speed']
                speed = speed * 5
                generated_images = vista_gen(image_path, speed)
                for cont in range(0, 25):
                    image_path_ = os.path.join(output_dir,
                                               os.path.basename(self.data_lists[start_index + cont]['Image File']))
                    to_pil = transforms.ToPILImage()
                    img = to_pil(generated_images[cont])
                    img.save(image_path_)
            sys.stdout = sys.__stdout__
            cleanup_pytorch_resources()

        results = test_ADS.get_violation(args,self.data_dir,os.path.join(self.data_dir, "original"),MR_dir,maneuver=self.maneuver)
        before_test = sum(item['violations'] for item in results)
        results = test_ADS.get_violation(args,self.data_dir,os.path.join(self.data_dir, "original"),output_dir,maneuver=self.maneuver)
        after_test = sum(item['violations'] for item in results)
        print(f"{type},before:{before_test},after:{after_test}")
        if type=="best":
            if after_test>before_test:
                MR_dir = output_dir
                self.bMR.append(self.MRs[after]["diffusion_prompt"])
        if type=="worst":
            if after_test < before_test:
                MR_dir = output_dir
                self.wMR.append(self.MRs[after]["diffusion_prompt"])

        return MR_dir

    def Iterative(self):
        under_test = self.data_process
        cache_file = os.path.join(self.save_place,'violation_1.npy')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                violation_tests = np.load(cache_file)
        else:
            violation_tests = self.get_violation(under_test)
            data_process.Check_file(self.save_place)
            np.save(cache_file, violation_tests)
        ############################################################
        paired = list(zip(under_test, violation_tests))
        sorted_pairs = sorted(paired, key=lambda x: x[1], reverse=True)
        sorted_under_test = [pair[0] for pair in sorted_pairs]
        Best_MR = sorted_under_test[len(sorted_under_test)//2-1]
        Worst_MR = sorted_under_test[-1]
        self.bMR.append(self.MRs[Best_MR]["diffusion_prompt"])
        self.wMR.append(self.MRs[Worst_MR]["diffusion_prompt"])
        bMR_dir = os.path.join(self.data_dir,"MRs",str(Best_MR))
        wMR_dir = os.path.join(self.data_dir,"MRs",str(Worst_MR))


        start_index = 0
        for iterative in range(self.test_round):
            match_best = sorted_under_test[len(sorted_under_test)//2-iterative-2]
            match_worst = sorted_under_test[len(sorted_under_test)-iterative-2]
            bMR_dir = self.apply_match(bMR_dir,match_best,type="best")
            wMR_dir = self.apply_match(wMR_dir, match_worst, type="worst")
        data_dict = {
            "bMR_dir": bMR_dir,
            "wMR_dir": wMR_dir,
            "bMR": self.bMR,
            "wMR": self.wMR
        }
        save_dir = os.path.join(self.data_dir, "saved_data")
        with open(os.path.join(self.data_dir, f"{self.Test_type}.json"), 'w') as f:
            json.dump(data_dict, f, indent=4)

    def final_result(self):

        under_test = self.data_process
        cache_file = os.path.join(self.save_place, 'violation_1.npy')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                violation_tests = np.load(cache_file)
        else:
            violation_tests = self.get_violation(under_test)
            data_process.Check_file(self.save_place)
            np.save(cache_file, violation_tests)
        ############################################################
        paired = list(zip(under_test, violation_tests))
        sorted_pairs = sorted(paired, key=lambda x: x[1], reverse=True)
        sorted_under_test = [pair[0] for pair in sorted_pairs]
        Best_MR = sorted_under_test[0]
        Worst_MR = sorted_under_test[-1]
        with open(os.path.join(self.data_dir, f"{self.Test_type}.json"), 'r') as file:
            data = json.load(file)
        bMR_dir = data['bMR_dir']
        wMR_dir = data['wMR_dir']
        bMR_dir_b = os.path.join(self.data_dir, "MRs", str(Best_MR))
        wMR_dir_b = os.path.join(self.data_dir, "MRs", str(Worst_MR))

        results = test_ADS.get_violation(args,self.data_dir,os.path.join(self.data_dir, "original"),bMR_dir_b,maneuver=self.maneuver)
        results_1 = test_ADS.get_violation(args,self.data_dir,os.path.join(self.data_dir, "original"),bMR_dir,maneuver=self.maneuver)
        print(f"source best:{results}\n,follow-up best:{results_1}")
        results = test_ADS.get_violation(args,self.data_dir,os.path.join(self.data_dir, "original"),wMR_dir_b,maneuver=self.maneuver)
        results_1 = test_ADS.get_violation(args,self.data_dir,os.path.join(self.data_dir, "original"),wMR_dir,maneuver=self.maneuver)
        print(f"source worst:{results}\n,follow-up worst:{results_1}")

    def step_result(self):
        subdirs = os.listdir(self.save_place)[:-1]
        violation_tests = []
        for i in range(len(subdirs)):
            results = test_ADS.get_violation(args, self.data_dir, os.path.join(self.data_dir, "original"),
                                             os.path.join(self.save_place, subdirs[i]),
                                             maneuver=self.maneuver)
            total_violations = sum(item['violations'] for item in results)
            violation_tests.append(total_violations)
        paired = list(zip(subdirs, violation_tests))
        sorted_pairs_1 = sorted(paired, key=lambda x: x[1], reverse=True)
        sorted_under_test_1 = [pair[0] for pair in sorted_pairs_1]
        under_test = self.data_process
        cache_file = os.path.join(self.save_place, 'violation_1.npy')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                violation_tests = np.load(cache_file)
        else:
            violation_tests = self.get_violation(under_test)
            data_process.Check_file(self.save_place)
            np.save(cache_file, violation_tests)
        ############################################################
        paired = list(zip(under_test, violation_tests))
        sorted_pairs = sorted(paired, key=lambda x: x[1], reverse=True)
        sorted_under_test = [pair[0] for pair in sorted_pairs]
        print(sorted_under_test,sorted_under_test_1)
        print(sorted_pairs, sorted_pairs_1)


#
class Get_Dataset_series(Dataset):
    def __init__(self, args,data_dir, save_path_, matched_data,size_control="resize"):
        self.data_dir = save_path_
        self.size_control = size_control
        self.matched_data = matched_data

    def __len__(self):
        return len(self.matched_data) - 5 + 1

    def __getitem__(self, idx):
        if any((i % 25 == 0) for i in range(idx + 1, idx + 5)):
            return None,None,None
        matched = self.matched_data.iloc[idx:idx + 5]

        img_sequence = []
        sequence = []
        prev_sequence = []
        cont = 0
        for _, row in matched.iterrows():
            data = [row['Steering Angle'],row['Vehicle Speed']]
            image_path = os.path.join(self.data_dir, os.path.basename(row['Image File']))
            if cont < 4:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                if self.size_control=="resize":
                    img = cv2.resize(img, (320, 160), interpolation=cv2.INTER_AREA)
                img = img.astype(numpy.float32) / 127.5 - 1.0
                img_sequence.append(img)
                prev_sequence.append(data)
            else:
                sequence.append(data)
            cont = cont + 1
        img_sequence = torch.from_numpy(numpy.array(img_sequence)).float().permute(0, 3, 1, 2)
        sequence = torch.tensor(sequence).float()
        prev_sequence = torch.tensor(prev_sequence).float()
        return img_sequence, prev_sequence, sequence


def Get_test_results(args,rule_number):
    input_dir = os.path.join(args.data_file, "results", "follow_up", "Exp_4",str(rule_number))


from imagecorruptions import get_corruption_names

if __name__ == "__main__":
    args = Paramenters.parse_args()
    #find_tests(args)

    slow_down_MRs = []
    #Exp  =EXP_Class_2(args)
   # Exp.Iterative()
    #Exp.step_result()
    Exp = EXP_Class_1(args, Test_type="Exp_1")
    Exp.step_result()
    Exp  = EXP_Class_1(args,Test_type="Exp_2")
    #Exp.Iterative()
    #Exp.final_result()
    Exp.step_result()
    """
    data_dir = os.path.join(args.data_file, "results", "follow_up", "Exp_4")
    pix2pix = MT.Pix2PixProcessor(strength=1)
    with open(os.path.join(data_dir, "info.json"), 'r', encoding='utf-8') as file:

        data_lists = json.load(file)
    for idx in range(0, len(data_lists),25):
         start_index = idx
         end_index = idx+25
         #image_path = os.path.join(MR_dir, os.path.basename(row['Image File']))
         MT.EXP_Pix2Pix(pix2pix, data_lists , os.path.join(data_dir,"original"), 0, os.path.join(data_dir,"MRs","8"), start_index, end_index,
                        "fog")"""
