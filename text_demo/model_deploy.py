#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:28:57 2019

@author: benjaminedwards
"""

from flask import Flask, request, render_template
import flask
import tensorflow as tf
import numpy as np
import keras
import pickle

app = Flask(__name__)


def load_model():
    global graph
    graph = tf.get_default_graph()
    global model
    model = keras.models.load_model('data/text_sentiment.h5')
    
    pkl_file = open('data/word_index.pkl', 'rb')
    global word_index
    word_index = pickle.load(pkl_file)
    pkl_file.close()    


def prep_data(new_text):
    words = keras.preprocessing.text.text_to_word_sequence(new_text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~   ', lower=True, split=' ')

    pred = np.array([word_index[word] if word in word_index else 0 for word in words])
    pred =[pred,[]]
    x_pred = keras.preprocessing.sequence.pad_sequences(pred, maxlen=100, padding='post')
    return x_pred


@app.route('/sentiment')
def my_form():
    return render_template('home.html')


@app.route('/sentiment', methods=["GET","POST"])
def my_form_get():
    data = {"success": False}
    new_text = request.form['text']
    x_pred = prep_data(new_text)
    
    with graph.as_default():
            data["prediction"] = model.predict(x_pred)[0][0]
            data["success"] = True

    # return a response in json format 
    return render_template('result.html',prediction = data["prediction"] )#return flask.jsonify(data) 
    

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
    		"please wait until server has fully started"))
    load_model()
    app.run(debug=True)



