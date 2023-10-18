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
  <img src="https://user-images.githubusercontent.com/97860488/221027525-6e94f1a6-ceb2-4aac-9ea8-de85c1d312c3.PNG" height= "200"/> <img src="https://user-images.githubusercontent.com/97860488/221027527-a28a3d89-a600-4375-9def-043eb58dd41a.PNG" height= "200"/>
 <img src="https://user-images.githubusercontent.com/97860488/221027509-fbf21184-b47b-405a-b738-b0e21af380d1.PNG" height= "200"/> <img src="https://user-images.githubusercontent.com/97860488/221027499-da13a9b1-4ab6-4b05-9d3f-cac1cbe3298e.PNG" height= "200"/>
</p>
