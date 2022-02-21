# Logo Detection 
Logo detection refers to detecting instances of logo of a certain class in images. The output from the logo detector describes the coordinates of the located logo in the image along with the class of the logo. These coordinates can be used to create boxes around the logos we have detected and label them correctly according to the class they belong to. A logo detection algorithm requires data to be provided in the form of images, locations of the logos in those images and the names of the logos for training the algorithm. 


![Text Detection example](https://libhub-readme.s3.us-west-2.amazonaws.com/vision/logo.jpg)
[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

# Recreate
This library is created to validate the input dataset format and split it into train/val/test datasets. The user needs to provide the path to two folders i.e the folder containing all the images and the folder containing all the label files.
> 📝 **Note**: For each image in the dataset there must be a corresponding label file of the same name in .txt format. For example if there is an image called **img1.png** in the images folder then there must a corresponding label file called **img1.txt** in the labels folder.
### Inputs
- `--images` path to the folder containing all the input images that contain the logos you want the detector to learn to detect.
- `--labels` path to the folder containing all the label files that contain the coordinates of the logos located in each image along with the class of each object.
Example of a label file is below:
    ```
    0 0.328125 0.17152777777777778 0.0671875 0.018055555555555554
    1 0.3953125 0.17083333333333334 0.0203125 0.022222222222222223
    2 0.305859375 0.22569444444444445 0.02734375 0.020833333333333332
    ```
    Each line represents the class and coordinates of a single logo located in the corresponding image. The fields are space delimited, and the coordinates are normalized from zero to one. To convert to normalized xywh from pixel values, divide center_x (and width) by the image's width and divide center_y (and height) by the image's height.
    ```
    class_id center_x center_y width height
    ```
    > 📝 **Note**: The numbering of class id has to start from zero and must be continous. 
    The order in which you provide number/class_id to the logos has to be kept in mind because when you provide the human readable names of each logo in the next library, they have to be in the same order. For example if you have 3 objects you want to train the detector on namely **Pepsi, Fanta and Sprite** and you provide class_id 0 as Pepsi, 1 as Fanta and 2 as Sprite in the label files, then the label names you provide in the next module must have the same order **"Pepsi, Fanta and Sprite"**
    
![Yolo format](https://libhub-readme.s3.us-west-2.amazonaws.com/vision/yolov5format.jpeg)
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


# Retrain
This library is used to retrain the neural network on the custom dataset. After the recreate library has been used to split the custom dataset into train/val/test splits, this library is used to retrain the neural network on the dataset provided.
As a result, we get a weight file that can be used in the future for detecting logos without having to retrain the neural network again.
### Flow
- The user has to upload the trainig dataset in the format of two directories. One directory containing all the images of the logos they want to train the detector on and another folder containing all the label files containing information about the logos boxes. This has been explained further in the documentation of **Recreate** module.
- The recreate module splits the dataset into train/val/test splits which serve as input to this retraining module.
- User can chose to specify the batch size, number of epochs and the names of classes/categories of the logos for training in this library.
- The class names represent the categories of the logos you want detect. For example, Pepsi, Fanta, Sprite etc. The class names can be provided directly as an argument or the user can chose to specify the location of the csv file containing the names of the classes. The csv file should contain class names under the **classes** column.

### Inputs
- `--batch` refers to the batch size that the neural network should ingest before calculatin the loss and readjusting the weights. For example if you have 100 images and a batch size of 2, the neural network will input two training images at a time, calculate loss, readjust the weights and then continue doing the same for the rest of the images in the training set.
- `--epochs` refers to the number of times the neural network will go through the entire training dataset to complete its' learning. The more the better but too high number can lead to overfitting and too low number can lead to underfitting.
- `--class_name`the names of the classes that you want your model to learn. Naming is important because otherwise the model would just be printing numbers like 0,1,2 etc. for the logos it has detected. So it is important we can map these numbers to meaningful names. You can provide class names as input in double quotes separated by commas for example, `"Pepsi, Fanta, Sprite"`.
    You may also provide input by specifying the csv file location. It would look something like this. `\dataset_name\input_file.csv` The csv file should have one column with name **classes** something like below:
    | classes |
    | :---:   |
    | Pepsi |
    | Fanta | 
    | Sprite |
    > 📝 **Note**: The order in which you provide the names has to be same as the order in which the items are present in the labels files. For example if in the label files you have provided labels 0,1,2 for Pepsi, Fanta, Sprite respectively, then the input to this parameter should also proceed in the same order "Pepsi, Fanta, Sprite" and the same order has to be preserved in the csv file.


### Outputs 
- `best.pt` refers to the file which contains the retrained neural network weights which yielded highest score while training. These can be used in the future for making inferences on images containing logos you finished retraining your neural network on. Download and save these weights.
 - `last.pt` refers to the file which contains the retrained neural network weights from the last epoch. These can be used in the future for making inferences on images containing logos you finished retraining your neural network on. We do not recommend using these weights for your logo detection task because they will perform worse than the **best** weights described above.
 - ` Trainresults.csv` refers to the file which contain the collection of all evaluation metrics evaluated on the training dataset. Images overall column contains the count of total training images, labels column contains the count of total labels available per class, mAp@.5 refers to mean average precision @ 0.5 and mAp@.5:.95 refers to mean average precision @ 0.5 to 0.95. All row contains the average of eval metrics from all classes. You may read more about eval metrics [here](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173).
      |classes | Images Overall | Labels | Precision | Recall | mAP@.5 | mAP@.5:.95 |
      | :---: | :-: | :-: | :-: | :-:| :---: | :-: | 
      | all | 701 | 986 | 0.93 | 0.89 | 0.92 | 0.93 | 
      | Pepsi | 701 | 626 | 0.94 | 0.89| 0.91 | 0.91 |
      | Fanta | 701 | 229 | 0.93 | 0.88| 0.92 | 0.92 |
      | Sprite | 701 | 131 | 0.92 | 0.90| 0.93 | 0.93 | 
- ` Valresults.csv` refers to the file which contain the collection of all evaluation metrics evaluated on the validation dataset. 
    classes | Images Overall | Labels | Precision | Recall | mAP@.5 | mAP@.5:.95 |
  | :---: | :-: | :-: | :-: | :-:| :---: | :-: | 
  | all | 100 | 186 | 0.916 | 0.873 | 0.904 | 0.912 | 
  | Pepsi | 100 | 86 | 0.90 | 0.89| 0.89 | 0.90 |
  | Fanta | 100 | 50 | 0.93 | 0.86| 0.914 | 0.92 |
  | Sprite | 100 | 50 | 0.92 | 0.87| 0.91 | 0.915 | 
- `others files` There are other output artifacts besides the ones mentioned above. These include results.csv file containing evaluations done every iteration, results.png contains plots for losses and evaluation metrics, a confusion matrix plot, plot for precision per class, plot for recall per class, plot for f1 score per class, plot for precision vs recall, plots for label distribution, plot for bounding box distribution, a yaml file containing training settings, a yaml file containing hyperparameters and there are 3 images containing examples of validation data, 3 images containing examples of training data and 3 images containing examples of predictions.
## How to run
```
cnvrg run  --datasets=<'[{id:Dataset Name,commit:Commit Id}]'> --machine=<Compute Size> --image=<Docker Image> --sync_before=<false> python3 train.py --batch <batch_size> --epochs <number_of_epochs> --class_name <class_names>
```
Example:
```
cnvrg run  --datasets='[{id:"logos",commit:"4cd66dfabbd964f8c6c4414b07cdb45dae692e19"}]' --machine="default.gpu-small" --image=tensorflow/tensorflow:latest-gpu --sync_before=false python3 train.py --batch 2 --epochs 100 --class_name "Pepsi, Fanta, Sprite"
```


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
Yolo (You only look once) is an object detection algorithm. The algorithm was introduced by Joseph Redmon et all in their 2015 [paper](https://arxiv.org/pdf/1506.02640.pdf) titled “You Only Look Once: Unified, Real-Time Object Detection”. It works by dividing the image into grid. Each block in the grid is responsible for detecting objects within itself. It is one of the fastest and most accurate algorithms for object detection peresent till date.
You can read more about the [history of yolo](https://machinelearningknowledge.ai/a-brief-history-of-yolo-object-detection-models/) and improvements made to it over the years.

# Reference
http://github.com/ultralytics/yolov5/
