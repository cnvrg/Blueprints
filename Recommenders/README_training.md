You can use this blueprint to train a custom model that can recommends similar items to users per their behaviours.
In order to train this model with your data, you would need to provide one folder located in s3:
- The folder needs to contain a file csv that includes 3 columns: user_id, item_id, rating (optional)
1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the following tasks to provide your data:

   In the `S3 Connector` task:
    * Under the `bucketname` parameter provide the bucket name of the data
    * Under the `prefix` parameter provide the main path to where the csv file is located

   In the `Data Validation` task:
    *  Under the `filename` parameter provide the path to the csv file including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>/<csv file>`

**NOTE**: You can use prebuilt data examples paths that are already provided

4. Click on the 'Run Flow' button
5. In a few minutes you will train a new recommender model and deploy as a new API endpoint
6. Go to the 'Serving' tab in the project and look for your endpoint
7. You can use the `Try it Live` section with relevant user id to get predictions
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have trained and deployed a custom model that can recommend similar to users per their behaviour!

[See here how we created this blueprint](https://github.com/cnvrg/Blueprints/tree/main/Recommenders)
