# Exploring options with flask apps and keras models.

## model_training.py
Using keras to train a bidirectional LSTM on the IMDB sentiment classifciation task.
Source: https://raw.githubusercontent.com/keras-team/keras/master/examples/imdb_bidirectional_lstm.py
Save the model.

## supporting_processing.py
pickle the imdb dictionary lookup for new new data.

## model_deploy.py
Creating a basic api with flask.
Preprocess the data with keras.preprocessing.
Return negative sentiment score.


### Example
visit - http://127.0.0.1:5000/sentiment
enter:"The film was terrible. Nicholas Cage was brilliant ofcourse, but the everything else was awful."

results: 
{
  "prediction": "0.8359641", 
  "success": true
}

