You can use this blueprint to train a tailored model that analyzes sentiment in text using your custom data.
In order to train this model with your data, you would need to provide a data of text sentences and sentiment pairs.
For your convenience, you can use one of Kaggle prebuilt datasets.
1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. 3. In the flow, edit the following tasks to provide your data:

In the `Kaggle Connector` task:
    * Under the `kaggle_username` parameter provide your kaggle username
    * Under the `kaggle_key` parameter provide your kaggle key
    * Under the `kaggle_dataset_name` choose the relevant dataset your would like to train your model with

   In the `Train` task:
    *  Under the `input_filename` parameter provide the path to the dataset,  it should look like:
       `/input/kaggle_connector/<csv file>`

**NOTE**: You can use prebuilt kaggle datasets that was already provided 

4. Click on the 'Run Flow' button
5. In a few minutes you will train a new sentiment analysis model and deploy as a new API endpoint
6. Go to the 'Serving' tab in the project and look for your endpoint
7. You can use the `Try it Live` section with any text to analyse the sentiment
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have trained and deployed a custom model that can analyse sentiment in text!

[See here how we created this blueprint](https://github.com/cnvrg/Blueprints/tree/main/Sentiment%20Analysis)
