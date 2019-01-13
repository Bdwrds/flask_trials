#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:45:17 2019

@author: benjaminedwards
"""


from keras.datasets import imdb
import pickle

word_index = imdb.get_word_index()


# write python dict to a file
output = open('word_index.pkl', 'wb')
pickle.dump(word_index, output)
output.close()



