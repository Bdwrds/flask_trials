
# Keras, meet Flask. 

Exploring the deployment of a keras model on flask.

This was entirely an exploratory piece to familiarise myself with a deployment process on flask with a pickled model.


### Installing

Install everything within the requirements.txt

```
pip install -r ./requirements.txt
```


## Deployment

### training_model.py
Trains and saves the keras lstm model. Also pickles the imdb dictionary lookup for new data.

Run training_model.py to train the bidirectional LSTM model on the IMDB sentiment data. 
```
python training_model.py
```

### model_deploy.py
Creating a very basic api with flask.
Preprocess the data with keras preprocessing.
Return a sentiment score.

Run deploy_model.py to run the app.
```
python deploy_model.py
```


### Example

Change to the app folder.
Run deploy_model.py and wait for the flask api to be ready.

Visit - http://127.0.0.1:5000/app

Enter something like:"The film was incredible! Nicholas Cage was brilliant. Its a shame Keira Knightley made an appearance.. but Cage saved the day! Loved it. Must see. Five Stars."

This will return a score indicating the sentiment of the text.

### Next Steps
TESTING. Variety of input text. 
Additional preprocessing of test data - match model input.


## Acknowledgments/sources

* Using keras to train a bidirectional LSTM on the IMDB sentiment classifciation task. Source: https://github.com/keras-team/keras/tree/master/examples


