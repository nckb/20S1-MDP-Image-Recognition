# 20S1-MDP-Image-Recognition

This repository details the resources and code utilized to perform the image recognition component for NTU CE/CZ3004 Multi Disciplinary Project, 2020/21 Semester 1, Group 32.

## Introduction

The RPi hosts a web server that serves a live MJPEG stream of the RPi's camera. Image recognition is performed in real-time on a PC using YOLOv4 Tiny. 

Although we had considered implementing features to determine the location and orientation of the detected images, there were limitations with our other existing code (Arduino & Algo) that did not make it a viable option. If you are considering implementing such features, it is recommended that you outline the specific requirements necessary across all teams **from the very start** in order to make it possible.


## Project dependencies

- [AlexeyAB/darknet](https://github.com/silvanmelchior/RPi_Cam_Web_Interface)
- [silvanmelchor/RPi_Cam_Web_Interface](https://github.com/silvanmelchior/RPi_Cam_Web_Interface)

Because the darknet project is ever-changing, a copy of the version used is included in this repository. Alternatively, you can use a more recent version to take advantage of newer improvements, but you may encounter compatibility issues.

As for RPi_Cam_Web_Interface, there should be no issues regardless of version. After all, we only need a MJPEG stream from it and nothing more.

## Prerequisites

 - **Python 3.7.9** (Python 3.8 modifies behavior for DLL imports)
 - CUDA 10.2
 - CuDNN 8.0.2 for CUDA 10.2
 - OpenCV 4.4.0
 - CMake 3.18.4

**You MUST use a PC with an NVIDIA GPU.**

Tested on a Dell XPS 13 running Windows 10. With an Intel 9th-gen Core i7 processor and an NVIDIA GeForce GTX 1050Ti, framerates in excess of 60fps were achieved.

## How to use

 **1. Install RPi_Cam_Web_Interface**

Configure the orientation and resolution of the video feed if necessary. Installation instructions can be found [here](https://elinux.org/RPi-Cam-Web-Interface).

 **2. Compile darknet**

The most time consuming part was to compile darknet. Following the recommended installation process using `vcpkg` turned out to be an exercise in futility. Instead, follow these two YouTube videos:

 - [YOLOv4 Tutorial #1 - Prerequisites for YOLOv4 Installation in 10 Steps](https://www.youtube.com/watch?v=5pYh1rFnNZs)
 - [YOLOv4 Object Detection Tutorial #2 - How to Run YOLOv4 on Images and video](https://www.youtube.com/watch?v=sUxAVpzZ8hU)

 **3. Train model and generate weights**

Take a sufficient number of pictures using the RPi camera. Use [LabelImg](https://github.com/tzutalin/labelImg) to draw bounding boxes and label each image. Try to use images taken at fixed distances, **within the arena and using the RPi camera only** to ensure accuracy of the weights. 

![Multiple images mounted on a block](https://github.com/nckb/20S1-MDP-Image-Recognition/blob/main/images/images-lab-labeled/multi_10.jpeg?raw=true)

To make things easier, you can attach multiple images to a block and swap them around.

Use this [Colab notebook](https://colab.research.google.com/drive/1PWOwg038EOGNddf6SXDG5AsC8PIcAe-G#scrollTo=Cdj4tmT5Cmdl) to train your weights (courtesy of Roboflow). They have a useful [tutorial](https://blog.roboflow.com/train-yolov4-tiny-on-custom-data-lighting-fast-detection/) if you want to learn more. Personally, I used Roboflow to prepare my dataset for training and it was a breeze to use. 

**Make sure you check on the progress of your Colab notebook regularly and save your trained weights! If the notebook times out, you lose ALL progress.**

Apart from the weights, you will also need to extract `./cfg/custom-yolov4-tiny-detector.cfg` and `./data/obj.names`.

**4. Test and configure to your liking**

Before running darknet, you need to tweak some files in order to get it working with your model and trained weights.

 - Copy your weights to the root directory.
 - Copy `custom-yolov4-tiny-detector.cfg` to `./cfg`.
 - Modify `./cfg/coco.data` to reflect the correct number of classes.
 - Replace `./data/coco.names` with the appropriate labels from `obj.names`.

To show a real-time image recognition stream with bounding boxes, use the following command:

`.\darknet detector demo ./cfg/coco.data ./cfg/custom_yolov4_tiny_detector.cfg  ./yolov4tiny.weights http://192.168.32.32/html/cam_pic_new.php`

Replace the IP address with the IP address of your RPi, and the locations of the relevant files. This mode is useful when you need to show the TA that the image recognition is working, or for testing 

`imgrec.py` contains the actual code used for the run. `imgrec_nonet.py` is the version without communication capabilities and may be useful for testing. However it is to be noted that **there is no penalty for false positives**, so there is no point to set a high threshhold for the actual run.
