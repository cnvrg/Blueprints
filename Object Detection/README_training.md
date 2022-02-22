You can use this blueprint to train a tailored model that detects objects in images using your custom data.
In order to train this model with your data, you would need to provide two folders located in s3:
- images: A folder with all the images you want to train the model
- labels: A folder with labels that correlates to the logos in the images folder
1. Click on "Use Blueprint" button
2. You will be redirected to your blueprint flow page
3. In the flow, click on the S3 connector and provide the paths to your images and labels folders, in example: "s3://<bucket>/<images-folder-path>"
4. Click on the 'Run Flow' button
5. In a few minutes you will train a new object detection model and deploy as a new API endpoint
6. Go to the 'Serving' tab in the project and look for your endpoint
7. You can use the "Try it Live" section with any image to check your model
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have deployed an API endpoint that detects objects in images!

[See here how we created this blueprint](https://github.com/cnvrg/Blueprints/tree/main/Object%20Detection)
