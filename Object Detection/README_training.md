You can use this blueprint to train a tailored model that detects objects in images using your custom data.
In order to train this model with your data, you would need to provide two folders located in s3:
- images: A folder with all the images you want to train the model
- labels: A folder with labels that correlates to the objects in the images folder
1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the following tasks to provide your data:

   In the `S3 Connector` task:
    * Under the `bucketname` parameter provide the bucket name of the data
    * Under the `prefix` parameter provide the main path to where the images and labels folders are located

   In the `Recreate` task:
    *  Under the `images` parameter provide the path to the images including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>/images`
    *  Under the `labels` parameter provide the path to the labels including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>/labels`

**NOTE**: You can use prebuilt data examples paths that are already provided

4. Click on the 'Run Flow' button
5. In a few minutes you will train a new object detection model and deploy as a new API endpoint
6. Go to the 'Serving' tab in the project and look for your endpoint
7. You can use the `Try it Live` section with any image to check your model
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have trained and deployed a custom model that detects objects in images!

[See here how we created this blueprint](https://github.com/cnvrg/Blueprints/tree/main/Object%20Detection)
