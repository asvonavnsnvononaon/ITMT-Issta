# An Iterative Traffic-Rule-Based Metamorphic Testing Framework for Autonomous Driving Systems
Introduction<br>
This repository is for the paper An Iterative Traffic-Rule-Based Metamorphic Testing Framework for Autonomous Driving Systems to ISSTA.<br>

<img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/overview%20of%20ITMT.png" width="60%"/>
 

This figure illustrates the workflow of ITMT, which consists of four main components. First, the LLM-based MR Generator automatically extracts MRs from traffic rules (Section 3.2). These MRs are expressed in Gherkin syntax to ensure structured and readable test scenarios. Second, the Test Scenario Generator creates continuous driving scenarios through image-to-image translation and diffusion-based image editing techniques (Section 3.3). Third, the Metamorphic Relations Validator verifies whether the model predictions satisfy the defined MRs (Section 3.4).<br>

1. The LLM-based MR Generator automatically extracts MRs from traffic rules use file EXP_1.py.<br>

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
Since EXP_2.py needs to process the dataset, we will describe in detail how to compose the dataset below.<br>

Here's how we organize our data format - simply place the corresponding driving scenario data into the appropriate folders as shown in the repository.<br>
This project is based on two well-known autonomous driving datasets: A2D2 and Udacity. To process the data, you first need to download these datasets:<br>

Udacity: <a href='https://github.com/udacity/self-driving-car?tab=readme-ov-file' target='_blank'>Self-Driving Car</a><br>
Notice: Due to privacy and data usage restrictions, we cannot provide direct download links for the Udacity dataset. Please obtain it through official channels.<br>
A2D2: <a href='https://www.a2d2.audi/a2d2/en.html' target='_blank'>A2D2</a><br>

For the Udacity dataset, we use HMB1, HMB2, HMB4, and HMB6 as training and validation sets, while HMB5 serves as the test set. For the A2D2 dataset, we combine data from Gaimersheim, Ingolstadt, and Munich regions, which are then proportionally split into training, validation, and test sets.<br>

After organizing the data, run main.py which includes the following steps:<br>

data_process.data_process(args) - Downsamples images and pairs sensor data<br>
OneFormer.Check_OneFormer(args) - Generates semantic segmentation results<br>
data_process.resize_images(args,"ORA") - Resizes images to 320x160 for training<br>
data_process.prepare_data(args) - Loads data into PyTorch structure<br>
train_ADS.Train(args) - Trains the autonomous driving system<br>

Coming soon.
