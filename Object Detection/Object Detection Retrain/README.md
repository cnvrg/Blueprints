# Object Detection 
Object detection refers to detecting instances of semantic objects of a certain class in images. The output from the object detector describes the coordinates of the located object in the image along with the class of the object. These coordinates can be used to create boxes around the objects we have detected and label them correctly according to the class they belong to. An object detection algorithm requires data to be provided in the form of images, locations of the objects in those images and the names of the object for training the algorithm. 

![Object detection example](https://libhub-readme.s3.us-west-2.amazonaws.com/vision/object.jfif)
[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

# Retrain
This library is used to retrain the neural network on the custom dataset. After the recreate library has been used to split the custom dataset into train/val/test splits, this library is used to retrain the neural network on the dataset provided.
As a result, we get a weight file that can be used in the future for detecting objects without having to retrain the neural network again.
### Flow
- The user has to upload the trainig dataset in the format of two directories. One directory containing all the images of the objects they want to train the detector on and another folder containing all the label files containing information about the bounding boxes. This has been explained further in the documentation of **Recreate** module.
- The recreate module splits the dataset into train/val/test splits which serve as input to this retraining module.
- User can chose to specify the batch size, number of epochs and the names of classes/categories of the objects for training in this library.
- The class names represent the categories of the objects you want detect. For example, person, dog, cat etc. The class names can be provided directly as an argument or the user can chose to specify the location of the csv file containing the names of the classes. The csv file should contain class names under the **classes** column.

### Inputs
- `--batch` refers to the batch size that the neural network should ingest before calculatin the loss and readjusting the weights. For example if you have 100 images and a batch size of 2, the neural network will input two training images at a time, calculate loss, readjust the weights and then continue doing the same for the rest of the images in the training set.
- `--epochs` refers to the number of times the neural network will go through the entire training dataset to complete its' learning. The more the better but too high number can lead to overfitting and too low number can lead to underfitting.
- `--class_name`the names of the classes that you want your model to learn. Naming is important because otherwise the model would just be printing numbers like 0,1,2 etc. for the objects it has detected. So it is important we can map these numbers to meaningful names. You can provide class names as input in double quotes separated by commas for example, `"Person,Dog,Cat"`.
    You may also provide input by specifying the csv file location. It would look something like this. `\dataset_name\input_file.csv` The csv file should have one column with name **classes** something like below:
    | classes |
    | :---:   |
    | Person |
    | Dog | 
    | Cat |
    > 📝 **Note**: The order in which you provide the names has to be same as the order in which the items are present in the labels files. For example if in the label files you have provided labels 0,1,2 for Person, Dog and Cat respectively, then the input to this parameter should also proceed in the same order "Person,Dog,Cat" and the same order has to be preserved in the csv file.


### Outputs 
- `best.pt` refers to the file which contains the retrained neural network weights which yielded highest score while training. These can be used in the future for making inferences on images containing objects you finished retraining your neural network on. Download and save these weights.
 - `last.pt` refers to the file which contains the retrained neural network weights from the last epoch. These can be used in the future for making inferences on images containing objects you finished retraining your neural network on. We do not recommend using these weights for your object detection task because they will perform worse than the **best** weights described above.
 - ` Trainresults.csv` refers to the file which contain the collection of all evaluation metrics evaluated on the training dataset. Images overall column contains the count of total training images, labels column contains the count of total labels available per class, mAp@.5 refers to mean average precision @ 0.5 and mAp@.5:.95 refers to mean average precision @ 0.5 to 0.95. All row contains the average of eval metrics from all classes. You may read more about eval metrics [here](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173).
      |classes | Images Overall | Labels | Precision | Recall | mAP@.5 | mAP@.5:.95 |
      | :---: | :-: | :-: | :-: | :-:| :---: | :-: | 
      | all | 701 | 986 | 0.93 | 0.89 | 0.92 | 0.93 | 
      | Person | 701 | 626 | 0.94 | 0.89| 0.91 | 0.91 |
      | Dog | 701 | 229 | 0.93 | 0.88| 0.92 | 0.92 |
      | Cat | 701 | 131 | 0.92 | 0.90| 0.93 | 0.93 | 
- ` Valresults.csv` refers to the file which contain the collection of all evaluation metrics evaluated on the validation dataset. 
    classes | Images Overall | Labels | Precision | Recall | mAP@.5 | mAP@.5:.95 |
  | :---: | :-: | :-: | :-: | :-:| :---: | :-: | 
  | all | 100 | 186 | 0.916 | 0.873 | 0.904 | 0.912 | 
  | Person | 100 | 86 | 0.90 | 0.89| 0.89 | 0.90 |
  | Dog | 100 | 50 | 0.93 | 0.86| 0.914 | 0.92 |
  | Cat | 100 | 50 | 0.92 | 0.87| 0.91 | 0.915 | 
- `others files` There are other output artifacts besides the ones mentioned above. These include results.csv file containing evaluations done every iteration, results.png contains plots for losses and evaluation metrics, a confusion matrix plot, plot for precision per class, plot for recall per class, plot for f1 score per class, plot for precision vs recall, plots for label distribution, plot for bounding box distribution, a yaml file containing training settings, a yaml file containing hyperparameters and there are 3 images containing examples of validation data, 3 images containing examples of training data and 3 images containing examples of predictions.
## How to run
```
cnvrg run  --datasets=<'[{id:Dataset Name,commit:Commit Id}]'> --machine=<Compute Size> --image=<Docker Image> --sync_before=<false> python3 train.py --batch <batch_size> --epochs <number_of_epochs> --class_name <class_names>
```
Example:
```
cnvrg run  --datasets='[{id:"object",commit:"4cd66dfabbd964f8c6c4414b07cdb45dae692e19"}]' --machine="default.gpu-small" --image=tensorflow/tensorflow:latest-gpu --sync_before=false python3 train.py --batch 2 --epochs 100 --class_name /input/s3_connector/object_detection_data/names.csv
```


# About Yolov5
Yolo stands for, You Only Look Once. It became famous because of its' speed and accuracy in detecting objects in images and videos. Every year since the algorithm has been modified and made better with a new version being released annually. We have implemented the latest of version of Yolov5 published and made available by [Ultralytics](https://github.com/ultralytics/yolov5). We have used the small model made available by Ultralytics. [Read more](https://pytorch.org/hub/ultralytics_yolov5/) about different model sizes.
Yolo (You only look once) is an object detection algorithm. The algorithm was introduced by Joseph Redmon et all in their 2015 [paper](https://arxiv.org/pdf/1506.02640.pdf) titled “You Only Look Once: Unified, Real-Time Object Detection”. It works by dividing the image into grid. Each block in the grid is responsible for detecting objects within itself. It is one of the fastest and most accurate algorithms for object detection peresent till date.
You can read more about the [history of yolo](https://machinelearningknowledge.ai/a-brief-history-of-yolo-object-detection-models/) and improvements made to it over the years.

# Reference
http://github.com/ultralytics/yolov5/