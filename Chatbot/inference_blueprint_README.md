# Inference
You can use this blueprint to immediately infer the intent of a message received from a costumer. For using this pretrained intent recognition model, you will need to create a ready-to-use API-endpoint that can be integrated with your data and application, in minutes.
This blueprint supports two modes:
1. `intent` -  returns the intent of the given text
2. `response` - returns a response message relevant to the identified intent based on a intent-response dictionary provided in the environment variable `responses_dict`
 You can the mode by defining intent/response in the `mode` environment variable.

Click on Use Blueprint button
In the pop up, choose the relevant compute you want to use to deploy your API endpoint
You will be redirected to your endpoint
You can use the Try it Live section with any text, i.e. "I forgot my password".
You can also integrate your API with your code using the integration panel at the bottom of the page
Congrats! You have deployed an API endpoint that analyse sentiment in text!
