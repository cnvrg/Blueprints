You can use this blueprint to train a tailored model that detects objects images and videos using your custom data.
In order to train this model with your data, you would need to provide two folders:
- images: A folder with all the images you want to train the model
- labels: A folder with labels that corraltes to the objects in the images folder
1. Click on "Use Blueprint" button
2. You will be redirected to your blueprint flow page
3. In the flow add a Data Task with the 2 folders specified above
4. Click on the 'Run Flow' button
5. In a few minutes you will train a new object detection model and deploy as a new API endpoint
6. Go to the 'Serving' tab in the project and look for your endpoint
7. You can now use the 'Try it Live' section to get results
8. You can also integrate your API with your code and your data


[See here how we created this blueprint](https://link-url-here.org)
