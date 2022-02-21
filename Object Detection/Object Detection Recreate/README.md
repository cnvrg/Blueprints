# Object Detection 
Object detection refers to detecting instances of semantic objects of a certain class in images. The output from the object detector describes the coordinates of the located object in the image along with the class of the object. These coordinates can be used to create boxes around the objects we have detected and label them correctly according to the class they belong to. An object detection algorithm requires data to be provided in the form of images, locations of the objects in those images and the names of the object for training the algorithm. 

![Object detection example](https://libhub-readme.s3.us-west-2.amazonaws.com/vision/object.jfif)
[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

# Recreate
This library is created to validate the input dataset format and split it into train/val/test datasets. The user needs to provide the path to two folders i.e the folder containing all the images and the folder containing all the label files.
> 📝 **Note**: For each image in the dataset there must be a corresponding label file of the same name in .txt format. For example if there is an image called **img1.png** in the images folder then there must a corresponding label file called **img1.txt** in the labels folder.
### Inputs
- `--images` path to the folder containing all the input images that contain the objects you want the detector to learn to detect.
- `--labels` path to the folder containing all the label files that contain the coordinates of the objects located in each image along with the class of each object.
Example of a label file is below:
    ```
    0 0.328125 0.17152777777777778 0.0671875 0.018055555555555554
    1 0.3953125 0.17083333333333334 0.0203125 0.022222222222222223
    2 0.305859375 0.22569444444444445 0.02734375 0.020833333333333332
    ```
    Each line represents the class and coordinates of a single object located in the corresponding image. The fields are space delimited, and the coordinates are normalized from zero to one. To convert to normalized xywh from pixel values, divide center_x (and width) by the image's width and divide center_y (and height) by the image's height.
    ```
    class_id center_x center_y width height
    ```
    > 📝 **Note**: The numbering of class id has to start from zero and must be continous. 
    The order in which you provide number/class_id to the object has to be kept in mind because when you provide the human readable names of each object in the next library, they have to be in the same order. For example if you have 3 objects you want to train the detector on namely **Person, Dog and Cat** and you provide class_id 0 as Person, 1 as Dog and 2 as Cat in the label files, then the label names you provide in the next module must have the same order **"Person, Dog, Cat"**
    
![yolo format](https://libhub-readme.s3.us-west-2.amazonaws.com/vision/yolov5format.jpeg)
### Outputs
The output containst he input dataset split into train/val/test datasets. The images and labels folders each contain three folders with names **train,test and val** which contain the images and labels. Following directory structure is created:
```
| - images
    | - train
        | - img13.jpg
        | - img24.jpg
        | ..
    | - val
        | - img73.jpg
        | - img30.jpg
        | ..
    | - test
        | - img2.jpg
        | - img50.jpg
        | ..
| - labels
    | - train
        | - img13.txt
        | - img24.txt
        | ..
    | - val
        | - img73.txt
        | - img30.txt
        | ..
    | - test
        | - img2.txt
        | - img50.txt
        | ..
```
## How to run
```
cnvrg run  --datasets=<'[{id:Dataset Name,commit:Commit Id}]'> --machine=<Compute Size> --image=<Docker Image> --sync_before=<false> python3 recreate.py --images <path to images folder> --labels <path to labels folder>
```
Example:
```
cnvrg run  --datasets='[{id:"logos",commit:"d54ad009d179ae346683cfc3603979bc99339ef7"}]' --machine="default.medium" --image=tensorflow/tensorflow:latest-gpu --sync_before=false python3 recreate.py --images /data/logos/images --labels /data/logos/labels
```


# About Yolov5
Yolo stands for, You Only Look Once. It became famous because of its' speed and accuracy in detecting objects in images and videos. Every year since the algorithm has been modified and made better with a new version being released annually. We have implemented the latest of version of Yolov5 published and made available by [Ultralytics](https://github.com/ultralytics/yolov5). We have used the small model made available by Ultralytics. [Read more](https://pytorch.org/hub/ultralytics_yolov5/) about different model sizes.
Yolo (You only look once) is an object detection algorithm. The algorithm was introduced by Joseph Redmon et all in their 2015 [paper](https://arxiv.org/pdf/1506.02640.pdf) titled “You Only Look Once: Unified, Real-Time Object Detection”. It works by dividing the image into grid. Each block in the grid is responsible for detecting objects within itself. It is one of the fastest and most accurate algorithms for object detection peresent till date.
You can read more about the [history of yolo](https://machinelearningknowledge.ai/a-brief-history-of-yolo-object-detection-models/) and improvements made to it over the years.

# Reference
http://github.com/ultralytics/yolov5/
