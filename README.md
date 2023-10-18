# 3D_Printing_Thesis_code

# Real Time Additive Manufacturing 3D Printer Vision Based Fault Detection System

To prepare the training dataset, a combination of an open-source dataset focused on 3D printing defects and additional images sourced from Google were utilized. This dataset consisted of a total of 257 images, encompassing various instances of 3D printing defects such as blobs, cracks, spaghetti, stringing, and under-extrusion.

# Data preparation

The annotation of images and the setup of the train-test split were accomplished using tools like VOTT and Roboflow. Subsequently, a repository leveraged CNN (Convolutional Neural Network) machine vision techniques, specifically YOLOv4 Tiny, to create a real-time fault detection model tailored for FDM (Fused Deposition Modeling) 3D printing applications.

# Data Training

Google Colab, a cloud-based platform, was utilized during the training process of the detection model. After 10,000 iterations, the model was trained using YOLOv4-Tiny's darknet architecture, resulting in the generation of configuration (cfg), weights, and name files for the model.

- Google Collab Training Notebook: https://colab.research.google.com/drive/17oU-cuJcYoama6hatZi_CREhWnAqBPbl?usp=sharing
  \*\*Dataset used is already attached to the notebook, can also be used to train other darknet based detection models.

## Phots Results

<p align="center"> 
  <img src="https://github.com/sulimanbadour1/3d_Printing_code_phd/blob/main/photos/1.JPG?raw=true" height= "200"/>
   <img src="https://github.com/sulimanbadour1/3d_Printing_code_phd/blob/main/photos/2.png?raw=true" height= "200"/>
 <img src="https://github.com/sulimanbadour1/3d_Printing_code_phd/blob/main/photos/3.png?raw=true" height= "200"/> 
</p>

<img src="https://raw.githubusercontent.com/sulimanbadour1/Sul_folio/main/src/assets/logos/white.png" width='100px'/>