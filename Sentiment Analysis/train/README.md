#  cnvrg sentiment_analysis_train

Notes for this Component

## How It Works

The library trains a sentiment analysis model on given labeled data and produces a prediction model and a tokenizer.
By default the library needs the receive a single path (--filename) for a local file.
The library splits the content of the file to train and test set 80% train and 20% test.


## How To Run

python3 sentiment_analysis.py --input_filename [YOUR_LABLED_DATA_FILE]

run python3 train.py -f  for info about more optional parameters
                                     
## Parameters

`--input_filename` - (String) (Required param) Path to a local labeled data file which contains the data that is used for training and validation.

`--output_token_file` - (String) (Default: 'tokenizer.pickle') Filename for saving the tokenizer.

`--output_model_file` - (String) (Default: 'model.h5') Filename for saving the model.

`--text_column` - (String) (Default: 'text') Name of text column in dataframe.

`--label_column` - (String) (Default: 'sentiment') Name of label column in dataframe.

`--epochs_val` - (int) (Default: 5) The number of epochs the algorithm performs in the training phaze.

`--batch_size_val` - (int) (Default: 32) The number of texts the model goes over in each epoch.