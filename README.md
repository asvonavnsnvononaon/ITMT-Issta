# An Iterative Traffic-Rule-Based Metamorphic Testing Framework for Autonomous Driving Systems
Introduction<br>
This repository is for the paper An Iterative Traffic-Rule-Based Metamorphic Testing Framework for Autonomous Driving Systems to ISSTA.<br>

<img src="https://github.com/asvonavnsnvononaon/ITMT-Issta/blob/main/Paper_images/overview%20of%20ITMT.png" width="60%"/>
 

This figure illustrates the workflow of ITMT, which consists of four main components. First, the LLM-based MR Generator automatically extracts MRs from traffic rules (Section 3.2). These MRs are expressed in Gherkin syntax to ensure structured and readable test scenarios. Second, the Test Scenario Generator creates continuous driving scenarios through image-to-image translation and diffusion-based image editing techniques (Section 3.3). Third, the Metamorphic Relations Validator verifies whether the model predictions satisfy the defined MRs (Section 3.4).<br>

1. The LLM-based MR Generator automatically extracts MRs from traffic rules use file EXP_1.py.

2. The Test Scenario Generator creates continuous driving scenarios through image-to-image translation and diffusion-based image editing techniques.
You can try 


