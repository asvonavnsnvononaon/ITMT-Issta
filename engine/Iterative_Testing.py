from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import engine.data_process



# process the image and text





inputs = processor.process(
    images=[Image.open("/media/yao/K/RMT_LLM/Data/results/follow_up/Exp_2_1/0/images/20180810150607_camera_frontcenter_000008050.png"),
            Image.open("/media/yao/K/RMT_LLM/Data/results/follow_up/Exp_2_1/0/images/20180810150607_camera_frontcenter_000008200.png"),],
    text="""Which picture best matches 'You are driving on a road approaching a crosswalk with pedestrians. The traffic signal ahead is flashing red.'? from 1-2? ."""
)

# move inputs to the correct device and make a batch of size 1
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# only get generated tokens; decode them to text
generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print(generated_text)

# >>>  This image features an adorable black Labrador puppy, captured from a top-down
#      perspective. The puppy is sitting on a wooden deck, which is composed ...
