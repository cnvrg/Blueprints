# Input
The OCR input will be an image in encoded base64 format 
# Output
The response from the endpoint will contain the text inside the image in plain text format, confidence scores associated with each detected text and the bounding box coordinates of each detection. The bounding box coordinates represent the coordinates of the center point, width of the box and height of the box.

`
center_x,
center_y, 
width, 
height`


An example json response from the endpoint is given below:
```
{
    "output": [
        {
            "text": "This is a sample text..",
            "conf": 0.75,
            "bbox": [
                80,
                150,
                75,
                40
            ]
        },
        {
            "class": "This is another sample line",
            "conf": 0.92,
            "bbox": [
                175,
                90,
                74,
                50
            ]
        }
    ]
}
```

Each response file contains a dictionary with key **"prediction"**.
The value for this key contains a list of all detections in the form of dictionaries. Each dictionary contains information about one single detection relating to its text, confidence score and bounding box coordinates.