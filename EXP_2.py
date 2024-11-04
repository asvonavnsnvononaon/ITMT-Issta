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
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from tqdm import tqdm
import numpy as np
import cv2
import imageio
from imagecorruptions import corrupt
def exp_2(args,save_path,pix=1):
    with open( os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    Pix_MRs = [mr for mr in MRs if mr["type"] == "Pix2Pix"]
    diffusion_MRs = [mr for mr in MRs if mr["type"] == "diffusion"]
    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)
    test_save_path = os.path.join(save_dir_2, save_path)
    with open(os.path.join(args.data_file, "results","info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)

# 186
    print(len(data_lists))#4650张图片
    args.pre_series = 25
    data_process.Check_file(os.path.join(test_save_path,"images",))
    data_process.Check_file(os.path.join(test_save_path,"videos",))
    data_process.Check_file(test_save_path)
    input_dir = save_dir
    mask_dir = save_dir_1
    output_dir = test_save_path
    ########################################1. Pix_to_Pix
    cont = 0

    if pix==1:
        pix2pix = MT.Pix2PixProcessor(strength=1)
        Pix_num = len(Pix_MRs)
        diffusion_num = len(diffusion_MRs)
        for i in range(Pix_num):
            output_dir = os.path.join(save_dir_2, save_path, str(i + diffusion_num), "images")
            output_dir_1 = os.path.join(save_dir_2, save_path,str(i+diffusion_num),"videos")

            data_process.Check_file(output_dir)
            data_process.Check_file(output_dir_1)
            diffusion_prompt = Pix_MRs[i]['diffusion_prompt']
            for idx in range(0, len(data_lists), args.pre_series):
                start_index = idx
                end_index = idx + args.pre_series
                MT.EXP_Pix2Pix(pix2pix, data_lists, input_dir, mask_dir, output_dir, start_index, start_index+1,
                               diffusion_prompt)
                MT.EXP_Pix2Pix(pix2pix, data_lists, input_dir,mask_dir, output_dir_1, start_index, end_index, diffusion_prompt)
    #diffusion
    if pix == 2:
        diffusion_num = len(diffusion_MRs)
        inpainter = MT.FluxInpainting(image_size=1024)
        args.pre_series=25
        for i in range(diffusion_num):
            diffusion_prompt = diffusion_MRs[i]['diffusion_prompt']
            test_save_path = os.path.join(save_dir_2, save_path,str(i))
            data_process.Check_file(os.path.join(test_save_path, "images", ))
            data_process.Check_file(os.path.join(test_save_path, "videos", ))
            output_dir = test_save_path

            for idx in tqdm(range(0, len(data_lists), args.pre_series)):
                start_index = idx
                end_index = idx + args.pre_series
                MT.EXP_diffusion(inpainter, data_lists, input_dir, mask_dir, output_dir, start_index, end_index,
                                 diffusion_prompt)

#"You are driving on a road approaching a crosswalk with pedestrians. The traffic signal ahead is flashing red.",

        #Vista
    if pix==3:
        gc.collect()
        torch.cuda.empty_cache()
        sys.path.append(os.path.join( "Other_tools", "Vista"))
        from my_test import VideoGenerator
        sys.stdout = open(os.devnull, 'w')

        vista_gen = VideoGenerator()
        diffusion_num = len(diffusion_MRs)
        for i in range(diffusion_num):
            diffusion_prompt = diffusion_MRs[i]['diffusion_prompt']
            test_save_path = os.path.join(save_dir_2, save_path, str(i))
            output_dir = test_save_path
            for idx in tqdm(range(0, len(data_lists), args.pre_series)):

                start_index = idx
                end_index = idx + args.pre_series
                MT.EXP_vista(args,vista_gen, data_lists, input_dir, mask_dir, output_dir, start_index, end_index, diffusion_prompt)

    if pix==4:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.enable_model_cpu_offload()
        for idx in range(0,  args.pre_series):
            start_index = idx
            end_index = idx + args.pre_series
            diffusion_prompt = diffusion_MRs[cont]['diffusion_prompt']
            MT.EXP_img2video(args, pipe, data_lists, input_dir, mask_dir, output_dir, start_index, end_index,
                         diffusion_prompt)


def exp_2_1(args, save_path):

    with open(os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    diffusion_MRs = [mr for mr in MRs if mr["type"] == "diffusion"]
    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)
    test_save_path = os.path.join(save_dir_2, save_path)
    with open(os.path.join(args.data_file, "results", "info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    args.pre_series = 25
    input_dir = save_dir
    mask_dir = save_dir_1
    output_dir = test_save_path
    ########################################1. Qwen
    diffusion_num = len(diffusion_MRs)
    batch_size = 1
    ##########################Load Qwen
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    for i in range(diffusion_num):
        output_dir = os.path.join(save_dir_2, save_path, str(i), "images")
        output_dir_2 = os.path.join(save_dir_2, save_path, str(i + diffusion_num), "Simages")
        data_process.Check_file(output_dir_2)
        diffusion_prompt = diffusion_MRs[i]['diffusion_prompt']
        img_list = os.listdir(output_dir)
        img_list.sort()
        text = "Describe this image, and analyze whether the elements in the image look most coherent, continuous, and without fragmentation."
        ALl_text=[]
        for i in range(0, len(img_list), batch_size):
            batch = img_list[i:i + batch_size]
            images = []
            for img_name in batch:
                img_path = os.path.join(output_dir, img_name)
                img = Image.open(img_path)
                images.append(img)
            inputs = processor.process(
                images=[Image.open(img_path)],
                text=text)
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):

                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            ALl_text.append(generated_text)
        save_batch_json(ALl_text, os.path.join(save_dir_2, save_path, str(i)))

def save_batch_json(generated_text, output_dir):
    json_filename = f"batch.json"
    json_path = os.path.join(output_dir, json_filename)

    data = {
        "generated_text": generated_text
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def Generate_videos(args,save_path):
    with open( os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    Pix_MRs = [mr for mr in MRs if mr["type"] == "Pix2Pix"]
    diffusion_MRs = [mr for mr in MRs if mr["type"] == "diffusion"]
    length = len(Pix_MRs)+len(diffusion_MRs)
    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)
    test_save_path = os.path.join(save_dir_2, save_path)
    with open(os.path.join(args.data_file, "results", "info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    args.pre_series = 25
    for i in range(length):
        image_dir = os.path.join(save_dir_2, save_path, str(i), "images")
        video_dir = os.path.join(save_dir_2, save_path, str(i), "videos")
        Exp_2_save = os.path.join(save_dir_2, save_path, str(i), "Exp")
        data_process.Check_file(Exp_2_save)
        cont = 0
        for idx in range(0, len(data_lists), args.pre_series):
            start_index = idx
            end_index = idx + args.pre_series

            image_path = os.path.join(image_dir, os.path.basename(data_lists[idx]['Image File']))
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, (640, 320), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(Exp_2_save, f"{cont}.png"), image_resized)
            frames=[]
            for idxx in range(start_index, end_index):
                image_path = os.path.join(video_dir, os.path.basename(data_lists[idxx]['Image File']))
                frame = cv2.imread(image_path)
                frame_resized = cv2.resize(frame, (320, 160), interpolation=cv2.INTER_AREA)
                frames.append(frame_resized)

            video_path = os.path.join(Exp_2_save, f"{cont}.mp4")
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for frame in frames:
                video.write(frame)
            video.release()
            cont += 1
def Generate_videos_1(args,save_path):
    with open( os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    Pix_MRs = [mr for mr in MRs if mr["type"] == "Pix2Pix"]
    diffusion_MRs = [mr for mr in MRs if mr["type"] == "diffusion"]
    length = len(Pix_MRs)+len(diffusion_MRs)
    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)
    test_save_path = os.path.join(save_dir_2, save_path)
    with open(os.path.join(args.data_file, "results", "info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    args.pre_series = 25
    input_dir = save_dir
    image_dir =input_dir
    Exp_2_save =  os.path.join(args.data_file, "results", "follow_up","Exp")
    data_process.Check_file(Exp_2_save)
    cont = 0
    for idx in range(0, len(data_lists), args.pre_series):
        start_index = idx
        end_index = idx + args.pre_series
        image_path = os.path.join(image_dir, os.path.basename(data_lists[idx]['Image File']))
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (640, 320), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(Exp_2_save, f"{cont}.png"), image_resized)
        frames = []
        for idxx in range(start_index, end_index):
            image_path = os.path.join(image_dir, os.path.basename(data_lists[idxx]['Image File']))
            frame = cv2.imread(image_path)
            frame_resized = cv2.resize(frame, (320, 160), interpolation=cv2.INTER_AREA)
            frames.append(frame_resized)
        video_path = os.path.join(Exp_2_save, f"{cont}.mp4")
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for frame in frames:
            video.write(frame)
        video.release()
        cont += 1

def exp_2_C(args, save_path):

    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)

    test_save_path = os.path.join(save_dir_2, save_path)
    with open(os.path.join(args.data_file, "results", "info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    corruption_MRs = [
        "gaussian_noise",
        "zoom_blur",
        "fog",
        "brightness",
        "pixelate"
    ]
    image_dir = save_dir
    severity=1
    for idx in tqdm(range(0, len(data_lists))):
        for i in range(len(corruption_MRs)):
            test_save_path = os.path.join(save_dir_2, save_path, str(i + 15))
            data_process.Check_file(test_save_path)
            image_path = os.path.join(image_dir, os.path.basename(data_lists[idx]['Image File']))
            img = Image.open(image_path)
            img = img.resize((320, 160), Image.LANCZOS)
            img = np.asarray(img)
            corrupted = corrupt(img, corruption_name=corruption_MRs[i], severity=severity + 1)
            corrupted = Image.fromarray(corrupted)


            corrupted.save(os.path.join(test_save_path, os.path.basename(data_lists[idx]['Image File'])))






def Generate_videos_x(args, save_path):
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

    # Define the gap between images (in pixels)
    gap = 20

    for i in range(length):
        image_dir = os.path.join(save_dir_2, save_path, str(i), "images")
        video_dir = os.path.join(save_dir_2, save_path, str(i), "videos")
        Exp_2_save = os.path.join(save_dir_2, save_path, str(i), "Exp")
        data_process.Check_file(Exp_2_save)
        cont = 0
        for idx in range(0, len(data_lists), args.pre_series):
            start_index = idx
            end_index = idx + args.pre_series

            # Read original and generated images
            original_image_path = os.path.join(save_dir, os.path.basename(data_lists[idx]['Image File']))
            generated_image_path = os.path.join(image_dir, os.path.basename(data_lists[idx]['Image File']))
            original_image = cv2.imread(original_image_path)
            generated_image = cv2.imread(generated_image_path)

            # Resize images
            original_image_resized = cv2.resize(original_image, (640, 320), interpolation=cv2.INTER_AREA)
            generated_image_resized = cv2.resize(generated_image, (640, 320), interpolation=cv2.INTER_AREA)

            # Create vertical comparison image with gap
            comparison_image = np.vstack((
                original_image_resized,
                np.full((gap, 640, 3), 255, dtype=np.uint8),  # White gap
                generated_image_resized
            ))
            cv2.imwrite(os.path.join(Exp_2_save, f"{cont}.png"), comparison_image)

            frames = []
            for idxx in range(start_index, end_index):
                original_frame_path = os.path.join(save_dir, os.path.basename(data_lists[idxx]['Image File']))
                generated_frame_path = os.path.join(video_dir, os.path.basename(data_lists[idxx]['Image File']))

                original_frame = cv2.imread(original_frame_path)
                generated_frame = cv2.imread(generated_frame_path)

                original_frame_resized = cv2.resize(original_frame, (640, 320), interpolation=cv2.INTER_AREA)
                generated_frame_resized = cv2.resize(generated_frame, (640, 320), interpolation=cv2.INTER_AREA)

                # Create vertical comparison frame with gap
                comparison_frame = np.vstack((
                    original_frame_resized,
                    np.full((gap, 640, 3), 255, dtype=np.uint8),  # White gap
                    generated_frame_resized
                ))
                frames.append(comparison_frame)

            video_path = os.path.join(Exp_2_save, f"{cont}.mp4")
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for frame in frames:
                video.write(frame)
            video.release()
            cont += 1



def Generate_gifs_X(args, save_path):
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

    # Define the gap between images (in pixels)
    gap = 10

    for i in range(length):
        image_dir = os.path.join(save_dir_2, save_path, str(i), "images")
        video_dir = os.path.join(save_dir_2, save_path, str(i), "videos")
        Exp_2_save = os.path.join(save_dir_2, save_path, str(i), "Exp_new")
        data_process.Check_file(Exp_2_save)
        cont = 0
        for idx in tqdm(range(0, len(data_lists), args.pre_series)):
            start_index = idx
            end_index = idx + args.pre_series

            # Read original and generated images
            original_image_path = os.path.join(save_dir, os.path.basename(data_lists[idx]['Image File']))
            generated_image_path = os.path.join(image_dir, os.path.basename(data_lists[idx]['Image File']))
            original_image = cv2.imread(original_image_path)
            generated_image = cv2.imread(generated_image_path)

            # Resize images to 320x160
            original_image_resized = cv2.resize(original_image, (320, 160), interpolation=cv2.INTER_AREA)
            generated_image_resized = cv2.resize(generated_image, (320, 160), interpolation=cv2.INTER_AREA)

            frames = []
            for idxx in range(start_index, start_index+ args.pre_series):
                original_frame_path = os.path.join(save_dir, os.path.basename(data_lists[idxx]['Image File']))
                generated_frame_path = os.path.join(video_dir, os.path.basename(data_lists[idxx]['Image File']))

                original_frame = cv2.imread(original_frame_path)
                generated_frame = cv2.imread(generated_frame_path)

                # Resize frames to 320x160
                original_frame_resized = cv2.resize(original_frame, (320, 160), interpolation=cv2.INTER_AREA)
                generated_frame_resized = cv2.resize(generated_frame, (320, 160), interpolation=cv2.INTER_AREA)

                # Create 4-part frame
                top = np.hstack((original_image_resized, original_frame_resized))
                bottom = np.hstack((generated_image_resized, generated_frame_resized))
                full_frame = np.vstack((top, np.full((gap, 640, 3), 255, dtype=np.uint8), bottom))  # White gap

                # Convert from BGR to RGB
                full_frame_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
                frames.append(full_frame_rgb)

            # Save as GIF
            gif_path = os.path.join(Exp_2_save, f"{cont}.gif")
            imageio.mimsave(gif_path, frames, duration=100,loop=0)
            cont += 1

if __name__ == "__main__":
    args = Paramenters.parse_args()
    #MT.find_test_images_1(args)

    #save_path="Exp_2_new"
    #exp_2(args,save_path)
    save_path = "Exp_2"
   # exp_2(args, save_path,pix=2)
    #exp_2(args, save_path, pix=3)
    exp_2_C(args, save_path)
   # Generate_videos_1(args, save_path)
    #Generate_gifs_X(args, save_path)


    #args.dataset =random.choice(Test_datasets)


    #Equality_MRs(args)
    #Test(args, "Pix2Pix",1)
    #Test(args,"diffusion",10) #add_object,add_object_1,Pix2Pix,diffusion
    #Test_ADS(args, type="Pix2Pix")
    #Test_ADS(args, type="diffusion")
