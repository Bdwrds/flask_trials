#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 10:44:11 2019

@author: benjaminedwards
"""



from keras.datasets import imdb
import numpy as np
import keras
import tensorflow as tf
word_index = imdb.get_word_index()

maxlen=100

global graph
graph = tf.get_default_graph()
model = keras.models.load_model('text_sentiment.h5')

model = keras.models.load_model('/Users/benjaminedwards/Documents/github_exp/flask_repo/text_demo/text_sentiment.h5')

new_text = "The movie was great, possibly the best film I've ever seen. terrible. wont see it again. dont recommend"

words = keras.preprocessing.text.text_to_word_sequence(new_text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~   ', lower=True, split=' ')

pred = np.array([word_index[word] if word in word_index else 0 for word in words])

pred =[pred,[]]

x_pred = keras.preprocessing.sequence.pad_sequences(pred, maxlen=maxlen, padding='post')

model.predict(x_pred)





import pickle

# write python dict to a file
output = open('word_index.pkl', 'wb')
pickle.dump(word_index, output)
output.close()
