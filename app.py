# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:34:45 2023

@author: beyen
"""

from flask import Flask, request, render_template
import pickle

from ast import literal_eval as le


# load category model
filename = 'Spam_Classification_best_model.pkl'
tuned_category_model = pickle.load(open(filename, 'rb'))

# load vectorizer
vector_filename = 'Spam_Classification_vectorizer.pkl'
vectorizer = pickle.load(open(vector_filename, 'rb'))

# load categories and class names
categories = {}

with open('category_labels.txt', 'r') as f:
    categories = le(f.read())


app = Flask(__name__)


def lookup(dictionary, value):
    """
    Get the key of a particular value in a dict.
    Input - Dictionary to map , Type: <dict>
    Output - key for the given value , Type: <str>
    """
    for k, v in dictionary.items():
        if v == value:
            return k
    return 'Not Found'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index.html')
def go_home():
    return render_template('index.html')


@app.route('/prediction.html')
def go_to_prediction():
    return render_template('prediction.html')


@app.route('/prediction', methods=['POST', 'GET'])
def predict(category_model=tuned_category_model, vectorizer=vectorizer, categories=categories):
    # get question from the html form
    text = request.form['message']

    # convert text to lower
    text = text.lower()

    # form feature vectors
    features = vectorizer.transform([text])

    # predict result category
    print('Using best category model: {}'.format(category_model))
    pred = category_model.predict(features)

    category = lookup(categories, pred[0])
    print('Category: {}'.format(category))

    return render_template('prediction.html', prediction_string='Prediction:', category='Category: {}'.format(category))


if __name__ == '__main__':
    app.run(debug=False)
