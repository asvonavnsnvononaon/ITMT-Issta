
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import engine.data_process
import os
import json
import torch
import engine.data_process as data_process
import engine.Paramenters as Paramenters
# Load the workbook
import transformers
args = Paramenters.parse_args()
from tqdm import tqdm

# load the processor
def Check_file(save_dir):
    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)
save_path = "Exp_2"

def save_batch_json(generated_text, output_dir):
    json_filename = f"batch.json"
    json_path = os.path.join(output_dir, json_filename)

    data = {
        "generated_text": generated_text
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def get_results_dir(args):
    save_dir = os.path.join(args.data_file, "results", "original")
    save_dir_2 = os.path.join(args.data_file, "results", "follow_up")
    save_dir_1 = os.path.join(args.data_file, "results", "mask")
    return save_dir,save_dir_1,save_dir_2

# load the mod
type=3

if type==1:
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0'
    )
    model.to(dtype=torch.bfloat16)

    with open(os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    diffusion_MRs = [mr for mr in MRs if mr["type"] == "diffusion"]
    save_dir, save_dir_1, save_dir_2 = get_results_dir(args)
    with open(os.path.join(args.data_file, "results", "info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    diffusion_num = len(diffusion_MRs)
    for i in range(diffusion_num):
        output_dir = os.path.join(save_dir_2, save_path, str(i), "images")
        text = "Describe this image, and analyze whether the elements in the image look most coherent, continuous, and without fragmentation."
        ALl_text = []
        img_list = os.listdir(output_dir)
        img_list.sort()
        for idx in tqdm(range(len(img_list))):

            inputs = processor.process(
                images=[Image.open(os.path.join(output_dir, img_list[idx]))],
                text=f"""prompt:{diffusion_MRs[i]}, return only   ,    "pass" or "not pass 
                with no additional text   please analyze: 
                Does the image description match the given prompt?" """
            )
            #"Describe this image, and analyze whether the elements in the image look most coherent, continuous, and without fragmentation."
            inputs["images"] = inputs["images"].to(torch.bfloat16)

            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=400, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            utput
            format: Return
            only
            "pass" or "not pass"
            with no additional text

            # print the generated text
            #print(generated_text)
            ALl_text.append(generated_text)
        save_batch_json(ALl_text, os.path.join(save_dir_2, save_path, str(i)))
if type==2:
    Pix_MRs = [mr for mr in MRs if mr["type"] == "Pix2Pix"]
    for i in range(len(Pix_MRs)):
        output_dir = os.path.join(save_dir_2, save_path, str(i+ diffusion_num), "images")
        text = "Describe this image, and analyze whether the elements in the image look most coherent, continuous, and without fragmentation."
        ALl_text = []
        img_list = os.listdir(output_dir)
        img_list.sort()
        for idx in tqdm(range(0,len(img_list))):
    #traffic rule->  changjing
            inputs = processor.process(
                images=[Image.open(os.path.join(output_dir, img_list[idx]))],
                text="Describe this image, and analyze whether the elements in the image look most coherent, continuous, and without fragmentation."
            )
            inputs["images"] = inputs["images"].to(torch.bfloat16)

            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # print the generated text
            #print(generated_text)
            ALl_text.append(generated_text)
        save_batch_json(ALl_text, os.path.join(save_dir_2, save_path, str(i+ diffusion_num)))


    for i in range(len(Pix_MRs)):
        output_dir = os.path.join(save_dir_2, save_path, str(i+ diffusion_num), "images")
        text = "Describe this image, and analyze whether the elements in the image look most coherent, continuous, and without fragmentation."
        ALl_text = []
        img_list = os.listdir(output_dir)
        img_list.sort()
        for idx in tqdm(range(0,len(img_list))):
    #traffic rule->  changjing
            inputs = processor.process(
                images=[Image.open(os.path.join(output_dir, img_list[idx]))],
                text="Describe this image, and analyze whether the elements in the image look most coherent, continuous, and without fragmentation."
            )
            inputs["images"] = inputs["images"].to(torch.bfloat16)

            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # print the generated text
            #print(generated_text)
            ALl_text.append(generated_text)
        save_batch_json(ALl_text, os.path.join(save_dir_2, save_path, str(i+ diffusion_num)))


if type==3:

    model_id = "arcee-ai/Llama-3.1-SuperNova-Lite"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto", )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]





    with open(os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    save_dir, save_dir_1, save_dir_2 = get_results_dir(args)
    diffusion_MRs = [mr for mr in MRs if mr["type"] == "diffusion"]
    for i in range(len(diffusion_MRs)):
        text = json.load(open(os.path.join(save_dir_2, save_path, str(i), "batch.json"), 'r', encoding='utf-8'))[
            "generated_text"]

        for j in range(len(text)):
            message = [{"role": "system",
                         "content": '''# CONTEXT #You are an expert in driving scene analysis.
                #Key Concepts# 1. Analyze if image content matches text description based on diffusion prompt and image understanding results
                2. Output format: Return only "pass" or "not pass" with no additional text'''},
        {"role": "user", "content": f" text description: {text[j]},prompt:{diffusion_MRs[i]}, please answer: Does text description support prompt"}]
            outputs = pipeline(
                message,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=pipeline.tokenizer.eos_token_id
            )
            answer = outputs[0]["generated_text"][-1]['content']
            print(answer)

"""
save_path = "Exp_2_1"
for i in range(diffusion_num):
    output_dir = os.path.join(save_dir_2, save_path, str(i), "images")
    output_dir_2 = os.path.join(save_dir_2, save_path, str(i + diffusion_num), "Simages")
    Check_file(output_dir_2)
    text = "Describe this image, and analyze whether the elements in the image look most coherent, continuous, and without fragmentation."
    ALl_text = []
    img_list = os.listdir(output_dir)
    img_list.sort()
    for idx in tqdm(range(len(img_list))):

        inputs = processor.process(
            images=[Image.open(os.path.join(output_dir, img_list[idx]))],
            text="Describe this image, and analyze whether the elements in the image look most coherent, continuous, and without fragmentation."
        )
        inputs["images"] = inputs["images"].to(torch.bfloat16)

        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        #print(generated_text)
        ALl_text.append(generated_text)
    save_batch_json(ALl_text, os.path.join(save_dir_2, save_path, str(i)))
"""
