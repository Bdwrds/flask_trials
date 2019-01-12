#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:58:58 2019

@author: benjaminedwards
"""


from flask import Flask
app_lala = Flask(__name__)

@app_lala.route('/hello_page_lala')
def hello_page_lulu():
    return('Hello Worldz!')

if __name__=="__main__":
    app_lala.run(debug=True)
    
    
