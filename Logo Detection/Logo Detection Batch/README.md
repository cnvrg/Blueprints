# Logo Detection 
Logo detection refers to detecting instances of logo of a certain class in images. The output from the logo detector describes the coordinates of the located logo in the image along with the class of the logo. These coordinates can be used to create boxes around the logos we have detected and label them correctly according to the class they belong to. A logo detection algorithm requires data to be provided in the form of images, locations of the logos in those images and the names of the logos for training the algorithm. 


![Text Detection example](https://libhub-readme.s3.us-west-2.amazonaws.com/vision/logo.jpg)
[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

# Detect
This library is used to make detections on input images provided. The input can be single image or a directory containing all the images you want to carry out logo detections. Along with the images, you would also need to provide the weight file if you want to carry logo detection on custom logos for which you trained your neural network using **Recreate** and **Retrain** libraries. **In case you want to detect default 3 logos namely `76, Steeden and Cafe Coffee Day` in your images, you don't need to specify the weights input.**
You will get the image or images as output with bounding box drawn around the logos that are detected.
### Input
- `source` provide the image path on which you want to carry out logo detections or provide the path to the folder containing all the images on which you want to carry out logo detections. 
- `weights` by default the weights are **logodetectdefault.pt** which will only detect the default 3 logos namely 76, Steeden and Cafe Coffee Day. In case you want to detect the custom logos you trained your neural network on in the **Retrain** library, you will have to provide the path to those weights. **Retrain** library outputs best.pt after training, download these and upload them to the dataset and provide the path for these weights.
- `conf` provide a input value ranging from 0 to 1. For every detection made by the logo detector there is a confidence score associated with it. This confidence score indicates how confident the logo detector is in its' prediction. For example if the detection has a confidence score of 0.25 it means that the logo detector is 25% confident that the bounding box in fact contains the logo it has detected. The input value in this parameter acts as a threshold and the final output will only contain logo detections for which the confidence score is above this threshold.
### Output
The output contains image/images contaning bounding boxes drawn around logos detected in the input images. Each box has a label written on top denoting the class of the logo and confidence score asscociated with that detection.
### How to run

```
cnvrg run  --datasets=<'[{id:Dataset Name,commit:Commit Id}]'> --machine=<Compute Size> --image=<Docker Image> --sync_before=<false> python3 detect.py --source <path to the image/images> --weights <path to the weights file> --conf <confidence threshold>
```
Example:
```
cnvrg run  --datasets='[{id:"logos",commit:"4cd66dfabbd964f8c6c4414b07cdb45dae692e19"}]' --machine="default.medium" --image=tensorflow/tensorflow:latest-gpu --sync_before=false python3 detect.py --source /data/logos/test --weights /data/logos/logoscnvrg.pt --conf 0.25
```

# About Yolov5
Yolo stands for, You Only Look Once. It became famous because of its' speed and accuracy in detecting objects in images and videos. Every year since the algorithm has been modified and made better with a new version being released annually. We have implemented the latest of version of Yolov5 published and made available by [Ultralytics](https://github.com/ultralytics/yolov5). We have used the small model made available by Ultralytics. [Read more](https://pytorch.org/hub/ultralytics_yolov5/) about different model sizes.
Yolo (You only look once) is an object detection algorithm. The algorithm was introduced by Joseph Redmon et all in their 2015 [paper](https://arxiv.org/pdf/1506.02640.pdf) titled ???You Only Look Once: Unified, Real-Time Object Detection???. It works by dividing the image into grid. Each block in the grid is responsible for detecting objects within itself. It is one of the fastest and most accurate algorithms for object detection peresent till date.
You can read more about the [history of yolo](https://machinelearningknowledge.ai/a-brief-history-of-yolo-object-detection-models/) and improvements made to it over the years.

# Reference
http://github.com/ultralytics/yolov5/
