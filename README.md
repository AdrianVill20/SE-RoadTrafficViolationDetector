# SE-RoadTrafficViolationDetector  

### Overview  
**SE-RoadTrafficViolationDetector** is an early-stage prototype that aims to detect traffic violations using artificial intelligence and real-time camera feeds. The system currently focuses on setting up the foundational environment, configuring dependencies, and initializing live object detection from a webcam.  

This stage serves as the groundwork for future development, where labeling and advanced classification (e.g., detecting cars, persons, or license plates) will be implemented next.  

---

###  Project Description  
The project’s ultimate goal is to automate the monitoring of road activity by:  
- Capturing real-time footage from cameras  
- Detecting vehicles and other objects  
- Preparing for future violation identification such as overspeeding, lane violations, or illegal turns  
- Collecting time, date, and vehicle details for later storage and reporting  

Currently, the project is configured to **initialize and display real-time object detection using OpenCV and TensorFlow**.  

---
##  Requirements
- **Python 3.10 or 3.11**

---


###  Installation & Setup  

#### 1. Clone or Download the Repository  
```bash
git clone https://github.com/AdrianVill20/SE-RoadTrafficViolationDetector.git
cd SE-RoadTrafficViolationDetector

```

#### 2. Create a Virtual Environment
It’s recommended to isolate dependencies using a virtual environment.
```bash
py -3.11 -m venv tfenv   
```
Or
```bash
py -3.10 -m venv tfenv   
```
### 3. Activate the Environment
Windows (PowerShell):
```bash
 .\tfenv\Scripts\Activate.ps1
```
macOS/Linux:
```bash
source tfenv/bin/activate
```
Check the python version
```bash
python --version
```
You should see Python 3.11.x or 3.10.x
### 4. Install Dependencies
Install the required libraries for TensorFlow and OpenCV.
```bash
python -m pip install --upgrade pip setuptools wheel
```
To install requirements go to directory cd src
```bash 
 pip install -r requirements.txt
```
(Optional but recommended:)
Verify the installations:
```bash
python -c "import cv2; print('OpenCV', cv2.__version__)"
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
python -c "import tensorflow_hub as hub; print('TensorFlow Hub', hub.__version__)"
```

## Running the Object Detection Script
Once dependencies are installed, run the demo file to start your camera stream:
```bash
python detect_demo.py
```
If successful, your webcam will open.<br>
The system currently detects general motion/objects.

---
## Optional Tests

### 1. Test Webcam Access
```bash
python webcam_test.py
```
If the window opens showing your camera feed, press Q to close it.

### 2. Test Basic Motion Detection
```bash
python motion_detector.py
```
This script should highlight moving objects in the camera feed.

## 🪖 Bike Helmet Detection using YOLOv8 and OpenCV

Bike helmets are essential for safety, but not everyone wears them. Traffic personnel often have difficulty monitoring every rider on the road. This project demonstrates how to automate helmet detection using Computer Vision and Deep Learning, specifically YOLOv8 and OpenCV.

This project includes:

Helmet detection in images

Real-time helmet detection in video or webcam feed

A trained YOLOv8 model using a custom dataset

## ⭐ Features

Detects whether a motorcycle rider is wearing a helmet

Works on images, video files, and webcams

Uses YOLOv8 (Ultralytics)

Simple Python implementation using OpenCV

## 📦 Requirements

Make sure you have Python 3.6+ installed.
Install all dependencies:

pip install gitpython>=3.1.30
pip install matplotlib>=3.3
pip install numpy>=1.23.5
pip install opencv-python>=4.1.1
pip install pillow>=10.3.0
pip install psutil
pip install PyYAML>=5.3.1
pip install requests>=2.32.0
pip install scipy>=1.4.1
pip install thop>=0.1.1
pip install torch>=1.8.0
pip install torchvision>=0.9.0
pip install tqdm>=4.64.0
pip install ultralytics>=8.2.34
pip install pandas>=1.1.4
pip install seaborn>=0.11.0
pip install setuptools>=65.5.1
pip install filterpy
pip install scikit-image
pip install lap

## 🛈 Training the YOLOv8 Model (Custom Dataset)
1. Download Dataset

Get the bike helmet dataset from Roboflow and unzip it.

2. Train on Google Colab

Open Colab → Runtime → Change runtime → Select T4 GPU

Verify GPU:

!nvidia-smi

Install Ultralytics:

!pip install ultralytics

3. Upload Dataset to Google Drive

My Drive/Datasets/BikeHelmet

Edit data.yaml to: ../drive/MyDrive/Datasets/BikeHelmet

4. Mount Google Drive

from google.colab import drive
drive.mount('/content/drive')

5. Train YOLOv8

!yolo task=detect mode=train model=yolov8l.pt data=../content/drive/MyDrive/Datasets/BikeHelmet/data.yaml epochs=100 imgsz=640

Training takes ~1–2 hours.

Download best.pt from:
runs/detect/train/weights/best.pt

## 📁 Project Folder Structure
BikeHelmetDetector/
├── Weights/
│   └── best.pt
├── Media/
│   └── riders_1.jpg
│   └── riders_2.jpg
│   └── riders_3.jpg
│   └── riders_4.jpg
│   └── riders_5.jpg
│   └── riders_6.jpg
├── helmet_detector.py

## ▶️ Running the Helmet Detector

Place your best.pt file in the Weights/ folder.

Run:

python helmet_detector.py

This script will:

Load your YOLO model

Run detection on images in the Media folder

Display bounding boxes showing riders and helmets

# Contributors

Christian Demetillo<br>
Shierwin Clark Niño<br>
Adrian Villarte
