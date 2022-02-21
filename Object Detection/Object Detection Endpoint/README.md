# End Point
# Output
The response from the endpoint will contain a list of detections in the form of, class of each logo detected, confidence score associated with each detection and the bounding box coordinates of each detection. The bounding box coordinates represent the coordinates of the center point, width of the box and height of the box.

`
center_x,
center_y, 
width, 
height`

![bounding box coordinates explained](https://libhub-readme.s3.us-west-2.amazonaws.com/vision/object-detection.png)

An example json response from the endpoint is given below:
```
{
    "output": [
        {
            "class": "Fanta",
            "conf": 0.65,
            "bbox": [
                100,
                150,
                50,
                50
            ]
        },
        {
            "class": "Sprite",
            "conf": 0.92,
            "bbox": [
                210,
                70,
                65,
                63
            ]
        }
    ]
}
```
![Json visualized](https://libhub-readme.s3.us-west-2.amazonaws.com/vision/object-detection-json.png)

Each response file contains a dictionary with key **"results"**.
The value for this key contains a list of all detections in the form of dictionaries. Each dictionary contains information about one single detection relating to its class, confidence score and bounding box coordinates.