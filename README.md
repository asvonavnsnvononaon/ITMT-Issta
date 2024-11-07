# An Iterative Traffic-Rule-Based Metamorphic Testing Framework for Autonomous Driving Systems
# 0. Introduction<br>
This repository is for the paper An Iterative Traffic-Rule-Based Metamorphic Testing Framework for Autonomous Driving Systems to ISSTA.<br>

<img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/overview%20of%20ITMT.png" width="60%"/>
 

This figure illustrates the workflow of ITMT, which consists of four main components. First, the LLM-based MR Generator automatically extracts MRs from traffic rules (Section 3.2). These MRs are expressed in Gherkin syntax to ensure structured and readable test scenarios. Second, the Test Scenario Generator creates continuous driving scenarios through image-to-image translation and diffusion-based image editing techniques (Section 3.3). Third, the Metamorphic Relations Validator verifies whether the model predictions satisfy the defined MRs (Section 3.4).<br>

# 1. MRs Generator
 The LLM-based MR Generator automatically extracts MRs from traffic rules use file <a href='https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/EXP_1.py' target='_blank'>EXP_1.py</a>.<br>

# 2.0 Using Open-Source Libraries as demo for Generating Driving Scenarios
2. The Test Scenario Generator creates continuous driving scenarios through image-to-image translation and diffusion-based image editing techniques use file EXP_2.py.<br>
(2.1) You can try Paper_images/test_image.png with prompt "You are driving on the road, add a pedestrian on the road" in <a href='https://huggingface.co/spaces/SkalskiP/FLUX.1-inpaint-dev' target='_blank'>FLUX.1 Inpaint Tool</a>.<br>
Note: As this demo doesn't support mask uploading, you'll need to manually select the mask area in the lower half of the image, including the road and vehicles.<br>
(2.2) You can then send this image to <a href='https://huggingface.co/spaces/rerun/Vista' target='_blank'>Vista</a> to generate test scenario.<br>
Note: 1.If the online demo is unavailable, we recommend using other text-to-video models such as <a href='https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD' target='_blank'>AnimateLCM-SVD</a> for generating driving scenarios. However, please be aware that models not specifically trained on driving datasets may generate videos with non-forward-facing perspectives.<br>
2.Due to GPU memory constraints, we use a parameter-reduced version of Vista that requires only 24GB VRAM. For potentially better performance, you can try the original Vista model at <a href='https://github.com/OpenDriveLab/Vista' target='_blank'>OpenDriveLab/Vista</a> (requires 40GB VRAM).<br>
(2.3) For environmental changes like weather conditions, we employ  <a href='https://huggingface.co/spaces/timbrooks/instruct-pix2pix' target='_blank'>InstructPix2Pix</a>. You can try Paper_images/test_image.png with prompt "turn this image  dusk" in this web.<br>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
    <img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/0.gif" width="30%">
    <img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/1.gif" width="30%">
    <img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/2.gif" width="30%">
    <img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/3.gif" width="30%">
    <img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/4.gif" width="30%">
    <img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/5.gif" width="30%">
    <img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/6.gif" width="30%">
    <img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/7.gif" width="30%">
    <img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/8.gif" width="30%">
</div>
<br>

# 2.1 Test Scenario Generator
## 2.1.1 Data Process 

Since EXP_2.py needs to process the dataset, we will describe in detail how to compose the dataset below.<br>

Here's how we organize our data format - simply place the corresponding driving scenario data into the appropriate folders as shown in the repository.<br>
This project is based on two well-known autonomous driving datasets: A2D2 and Udacity. To process the data, you first need to download these datasets:<br>

Udacity: <a href='https://github.com/udacity/self-driving-car?tab=readme-ov-file' target='_blank'>Self-Driving Car</a><br>
Notice: Due to privacy and data usage restrictions, we cannot provide direct download links for the Udacity dataset. Please obtain it through official channels.<br>
A2D2: <a href='https://www.a2d2.audi/a2d2/en.html' target='_blank'>A2D2</a><br>

For the Udacity dataset, we use HMB1, HMB2, HMB4, and HMB6 as training and validation sets, while HMB5 serves as the test set. For the A2D2 dataset, we combine data from Gaimersheim, Ingolstadt, and Munich regions, which are then proportionally split into training, validation, and test sets.<br>

After organizing the data, run <a href='https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/main.py' target='_blank'>main.py</a>.<br> which includes the following steps:<br>

data_process.data_process(args) - Downsamples images and pairs sensor data<br>
OneFormer.Check_OneFormer(args) - Generates semantic segmentation results<br>
data_process.resize_images(args,"ORA") - Resizes images to 320x160 for training<br>
data_process.prepare_data(args) - Loads data into PyTorch structure<br>
train_ADS.Train(args) - Trains the autonomous driving system<br>



## 2.1.2 Obtaining Test Set from Two Datasets
run <a href='https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/EXP_2.py' target='_blank'>EXP_2.py</a>.<br> 
MT.find_test_images_1(args) - Obtaining test set from two datasets

## 2.1.3 Generating Driving Scenarios
During the generation process, we organize tasks based on the diffusion techniques used rather than following the MR sequence. For example, we first batch process all scenarios requiring environmental changes. This approach offers two key advantages:<br>

1.It requires loading each diffusion model only once, optimizing memory usage. It prevents potential VRAM overflow that could occur from repeatedly loading different diffusion models <br>
3.It significantly improves generation speed by reducing model loading time<br>
We organize our generation functions as follows:<br>

exp_2(args,save_path,pix=1) - Generates driving scenario images through image-inpainting<br>
exp_2(args,save_path,pix=2) - Generates environmental variations using InstructPix2Pix<br>
exp_2(args,save_path,pix=3) - Creates scenario videos from generated images using Vista<br>

Additionally, we provide two more functions for scenario generation:<br>

Generate_gifs_X(args, save_path) - Creates GIF animations from generated sequences<br>
exp_2_C(args, save_path) - Simulates common driving conditions from DeepTest (e.g., blurring, rain, fog, etc.) through image corruption<br>

## 2.1.4 Vision LLM Quality Assessment
Run with different evaluation modes:<br>
Set type=2 to get vision LLM evaluation results,Set type=3 to use LLM to analyze results<br>
Advanced Analysis Recommendation:<br>
SoTA LLMs can provide more accurate analysis results. Due to token limitations in analyzing large numbers of scenarios, we recommend using web-based LLMs (such as ChatGPT, Claude) for better analysis. Input vision LLM results and use the following prompt:<br>
Please find the number of scenarios that not match the metamorphic testing prompt .... Ensure the discribe of image quality is good. Only provide the scenario numbers.
Coming soon.
