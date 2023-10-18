<img src="https://raw.githubusercontent.com/sulimanbadour1/Sul_folio/main/src/assets/logos/white.png" width='100px'/>

# 3D_Printing_Thesis_code

# Real Time Additive Manufacturing 3D Printer Vision Based Fault Detection System

To prepare the training dataset, a combination of an open-source dataset focused on 3D printing defects and additional images sourced from Google were utilized. This dataset consisted of a total of 257 images, encompassing various instances of 3D printing defects such as blobs, cracks, spaghetti, stringing, and under-extrusion.

# Data preparation

The annotation of images and the setup of the train-test split were accomplished using tools like VOTT and Roboflow. Subsequently, a repository leveraged CNN (Convolutional Neural Network) machine vision techniques, specifically YOLOv4 Tiny, to create a real-time fault detection model tailored for FDM (Fused Deposition Modeling) 3D printing applications.

# Data Training

The detection model was trained using Google Colab, a cloud-based platform, through 10,000 iterations. This training process utilized the darknet architecture of YOLOv4-Tiny, leading to the creation of configuration (cfg), weights, and name files for the model.

- Google Collab Training Notebook: https://colab.research.google.com/drive/17oU-cuJcYoama6hatZi_CREhWnAqBPbl?usp=sharing

- The attached dataset in the notebook can be utilized for training other detection models based on Darknet.

## Phots Results

After initializing YOLO files, the camera feed is presented, and in case of any detected faults, the feed will show a bounding box along with its associated class. While these bounding boxes are being displayed, the detection results are simultaneously logged into a text file. This text file not only contains the detection outcomes but also includes details about the session when the program was started, including the date and time when the program script was executed.

<p align="center"> 
  <img src="https://github.com/sulimanbadour1/3d_Printing_code_phd/blob/main/photos/1.JPG?raw=true" height= "200"/>
   <img src="https://github.com/sulimanbadour1/3d_Printing_code_phd/blob/main/photos/2.png?raw=true" height= "200"/>
 <img src="https://github.com/sulimanbadour1/3d_Printing_code_phd/blob/main/photos/3.png?raw=true" height= "200"/> 
</p>
