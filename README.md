# ADAS for PM
**Advanced Driver Assistance System for Personal Mobility**
- Model: YOLOv5 <br>
git clone https://github.com/ultralytics/yolov5
- Object Tracking Algorithm: Deep SORT
- Demo App: Swift
## Table of Contents

1. [Introduction](#introduction)
    - 1.1 [Need for Driver Assistance System](#need-for-driver-assistance-system)
    - 1.2 [Purpose](#purpose)
2. [Project Pipeline](#project-pipeline)
    - 2.1 [Data Collection](#data-collection)
    - 2.2 [Dataset Construction](#dataset-construction)
    - 2.3 [Model Training](#model-training)
    - 2.4 [Mobile Deployment](#mobile-deployment)
    - 2.5 [Testing](#testing)
3. [Procedure](#procedure)
    - 3.1 [Training Dataset](#training-dataset)
    - 3.2 [Dataset Processing](#dataset-processing)
        - 3.2.1 [Decompressing Dataset](#decompressing-dataset)
        - 3.2.2 [XML to CSV Conversion](#xml-to-csv-conversion)
        - 3.2.3 [CSV to TXT Conversion (Final Label File)](#csv-to-txt-conversion-final-label-file)
    - 3.3 [Model Training Process](#model-training-process)
    - 3.4 [Object Speed Estimation](#object-speed-estimation)
    - 3.5 [Application Testing](#application-testing)
        - 3.5.1 [Test Experiment](#test-experiment)
        - 3.5.2 [Discussion and Future Work](#discussion-and-future-work)

---

## 1. Introduction

### 1.1 Need for Driver Assistance System

In recent years, there has been a significant increase in accidents involving Personal Mobility (PM) devices. According to the Korea Road Traffic Authority (April 2023), there were **3,421 PM-related accidents** between 2017 and 2021, resulting in 45 fatalities. Additionally, **26 fatalities** were reported in 2022 alone.

The growing popularity of PM devices for short-distance travel has also increased the risk of accidents. This project proposes a **driver assistance system** designed to reduce these risks by analyzing potential hazards in the operating environment of PM devices.

### 1.2 Purpose

The project aims to:
- Detect obstacles and hazards during PM operation.
- Estimate the speed of approaching objects to provide timely collision warnings.

---

## 2. Project Pipeline

### 2.1 Data Collection

The project uses **Pedestrian Crossing Videos** from AI Hub, containing **29 object classes** related to collision risks during pedestrian movement. The data includes bounding boxes for moving and stationary objects, and polygon masks for road obstacles like manholes, cracks, and speed bumps.

### 2.2 Dataset Construction

- The dataset, sized at **300GB**, was decompressed and reorganized for training.
- XML files were used to extract the necessary data and create label files.

### 2.3 Model Training

The **YOLOv5 model** was selected for training based on factors such as model size, performance, and availability. Training was conducted using **Google Colab** with a **v100 GPU** and **PyTorch**.

- The final dataset consisted of **200,000 images** for training and **20,000 images** for validation.

### 2.4 Mobile Deployment

The trained model was converted to **PyTorch Lite** for mobile deployment. The bounding box data was used to calculate risk levels based on object speed estimation. Two algorithms were considered:
1. **SORT**
2. A custom algorithm based on bounding box size.

### 2.5 Testing

The application was tested by deploying it on an **iOS device** and installing it on an electric scooter for real-world testing.

---

## 3. Procedure
## 3.1 Training Dataset

![image](https://github.com/user-attachments/assets/cabbafad-5503-400f-843b-7f90e3ede84f)

The dataset consists of annotations in both bounding box and polygon forms for 29 types of objects that pose collision risks during pedestrian movement, such as cars, pedestrians, and utility poles. Additionally, the dataset includes polygon annotations for road conditions during pedestrian movement, such as speed bumps and manholes.

The obstacles are classified into several categories, and since the dataset covers most potential risk factors encountered during personal mobility device operation, we chose it for this project.

![image](https://github.com/user-attachments/assets/c7edea90-108e-4b59-8eb1-cd1b976eee78)


Results from the Exploratory Data Analysis (EDA) on the training dataset showed the number of bounding boxes per object. It was observed that the data for the "scooter" class, representing personal mobility devices, was relatively scarce compared to the "car" and "person" classes. This noticeable class imbalance, particularly the shortage of data for "scooters," indicated a need for data augmentation. A separate data folder was created for the "scooter" class, and augmentation was applied to mitigate this imbalance.
![image](https://github.com/user-attachments/assets/74e9e06c-89b4-478b-83ee-18e8ed9518f8)


---

## 3.2 Dataset Construction

The dataset size was 300GB, so a significant amount of time was spent decompressing and restructuring directories after extraction. Additionally, using Google Drive presented challenges, as having too much data in one folder caused Input/Output Errors when accessing the drive, hindering the training process.

To address these issues, we constructed a custom data preprocessing pipeline. This pipeline enabled efficient organization and preprocessing of the data, overcoming the initial challenges.

### 3.2.1 Decompressing the Dataset
The first step was decompressing the large dataset.
![image](https://github.com/user-attachments/assets/807e984a-488d-49a7-8b46-f46fb9c3aeb1)

### 3.2.2 Creating CSV Files from XML Files
After unzipping, the XML data was converted into CSV files for analysis, and TXT files were created for labels during model training. The meta-information and annotations for the image data were provided in XML format. To use this data for model training, it was necessary to convert it into TXT format. Only the values needed for the final labels were extracted and saved in a TXT file. Initially, CSV files were used to easily analyze the distribution of objects in the training data.

### 3.2.3 Creating TXT Files from CSV Files
Following the YOLO format, we extracted the class ID, and the relative x and y coordinates of the bounding box center, along with the bounding box's width and height. These values were normalized by dividing them by the image's width and height, adhering to the YOLO format.

---

## 3.3 Model Training

### 3.3.1 Attempt 1
During normalization of the bounding box coordinates, values exceeding 1 were discovered. The dataset description stated that the width and height of the images were 1920x1080, but some images had dimensions exceeding 3000x2000. To address this, we revised the normalization process to use the actual dimensions from the CSV file instead of fixed constants.

### 3.3.2 Attempt 2
In an attempt to speed up the loading of attributes from XML files, we retrieved all values at once. However, this caused errors because the order of attributes varied between files. We initially assumed a fixed attribute order (label, occluded, x1, y1, x2, y2), but the actual data showed variations. Though slower, the second attempt ensured that all attributes were matched correctly.

### 3.3.3 Attempt 3 - Final Version
Initially, we used YOLOv5's default parameters. However, the large training dataset caused frequent disconnections in the Google Colab runtime during data loading. We also found that unnecessary data augmentation was applied to the entire dataset. To address this, a separate data augmentation folder was created and applied only where necessary, avoiding augmentation for the entire dataset.

![image](https://github.com/user-attachments/assets/aae2fd6e-5e84-497b-87d8-ee2a5a427f90)

---

## 3.4 Speed Estimation Algorithm Development

### 3.4.1 Loading YOLO Files for Mobile Deployment
We expected converting YOLO files to PyTorch Lite (PTL) for mobile environments to be straightforward, based on official tutorials. However, it took much longer due to challenges in adapting the code for our system and Swift environment. Debugging was particularly difficult as Swift was new to our team. We later discovered the issue stemmed from version mismatches between Python and PyTorch. By downgrading both versions, we successfully ran the YOLO mobile code.

### 3.4.2 Speed Estimation Algorithm

#### Converting DeepSort Weight Files
Once the YOLO files were converted for mobile use, we attempted to load the YOLO model via the DeepSort detector in Swift. However, DeepSort’s weight files were in h7 format, not PT format, requiring a different PyTorch version (0.4.1). Due to delays in setting up the environment, we explored an alternative approach.

#### Implementing the Sort Algorithm
To avoid errors while loading the model weight files, we implemented the Sort algorithm directly in Swift. Initially, the code used both Objective-C and Swift. When adapting this to our system, we encountered issues with structure conversion errors and data types. Since the necessary BridgeHeader for using Objective-C classes in Swift was not recognized, this approach became impractical. Ultimately, we implemented the Sort algorithm, along with related components like the KalmanFilter, from scratch in Swift.

#### Implementing Custom Algorithm
While we initially thought measuring risk based solely on bounding box size wouldn’t be very accurate, it surprisingly outperformed the Sort algorithm in some cases. The Sort algorithm often failed to recognize the same object across consecutive frames, occasionally failing to trigger warnings in dangerous situations. Adjusting the algorithm to sound warnings based on bounding box size and screen position proved more effective. Consequently, the final implementation considered both bounding box size and the distance from the bottom of the screen to identify risky objects and trigger warnings.

---

## 3.5 Testing

### 3.5.1 Real-World Testing
You can view the actual test videos here:
- [YouTube Video](https://www.youtube.com/watch?v=zf_GswtFD6k)

<img src="https://github.com/user-attachments/assets/74364f2d-9bd1-4482-8297-9476c1222b5e" alt="image1" width="400"/>
<img src="https://github.com/user-attachments/assets/f616cb10-0fba-49d7-b69b-77139055a31a" alt="image2" width="400"/>



---

## 3.5.2 Discussion and Future Plans

### YOLO Model Accuracy and Inference Speed
In real-world driving scenarios, the model sometimes recognized objects too slowly, causing bounding boxes to appear over empty spaces. Adjusting the model for mobile deployment likely caused a slight reduction in both inference speed and accuracy. More precise speed estimation requires better object size measurements, and improving accuracy could be achieved by extending training time on a GPU.

### Speed Estimation Algorithm Improvement
The initially planned depth estimation method is generally used for larger objects, where a 1-meter error is not significant. However, for personal mobility devices and pedestrians, much higher accuracy is needed. Object tracking seems to be a better method, but implementing it on mobile devices presents challenges due to limited reference materials. Future work will focus on integrating object tracking into the current algorithm to create a more robust solution.
