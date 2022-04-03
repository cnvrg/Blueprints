# Chatbot Train
A chatbot is a computer program that simulates and processes human conversation, allowing humans to interact with digital devices as if they were communicating with a real person.
This chatbots main goal is to help businesses communicate with their costumers without requiring human resources.

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

# Retrain
This library is used to retrain the neural network on the custom dataset.
As a result, we get a model file that can be used to build a chatbot that fits the business's needs and can detect the costumers' intents and respond accordingly.
### Flow
- The user has to upload the trainig dataset which is a collection of possible messages mapped to their intent. The dataset should be in the format of a csv file containing two columns; the first is the intent and the second is the text. 
- The model is trained on the extended dataset and a model file is produced. The model can then be used for personalized business chatbot.

### Inputs
- `--data` refers to the base training dataset.
- `--additional_data` refers to the nadditional training dataset that contains the data that's unique to the business.
- `--model_file` refers to the model's file name.

### Outputs 
- `chatbot_model.pt` refers to the file which contains the retrained model. This can be used in the future for detecting the intent of a costumers message.
 
## How to run
```
python3 train.py -data <name of data file>
```
Example:
```
python3 train.py -data 'data.csv'
```


# About BERT
BERT is a method of pre-training language representations, meaning that we train a general-purpose "language understanding" model on a large text corpus (like Wikipedia), and then use that model for downstream NLP tasks that we care about (like question answering). BERT outperforms previous methods because it is the first unsupervised, deeply bidirectional system for pre-training NLP.

# Reference
https://github.com/google-research/bert