from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model checkpoint
model_checkpoint = "gokaygokay/Flux-Prompt-Enhance"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

enhancer = pipeline('text2text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    repetition_penalty= 1.2,
                    device=device)

max_target_length = 75
prefix = "enhance prompt: "


import random

short_prompt = "intersection, a pedestrian ahead,  a school bus ahead."
answer = enhancer(prefix + short_prompt, max_length=max_target_length)
final_answer = answer[0]['generated_text']

print(final_answer)
"""
with open('Texas_final.json', 'r', encoding='utf-8') as file:
    data_all = json.load(file)
processed_data = []
random_sample = random.sample(data_all, 10)

ii =0
for data in random_sample:
    ii+=1

    short_prompt = data["diffusion_prompt"]
    answer = enhancer(prefix + short_prompt, max_length=max_target_length)
    final_answer = answer[0]['generated_text']
    data_ = {
        "prompt": data["prompt"],
        "diffusion_prompt": final_answer
    }
    processed_data.append(data_)
with open(f"Texas_final_1.json", 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)
"""