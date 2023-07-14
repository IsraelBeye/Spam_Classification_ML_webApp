# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:26:59 2023

@author: Israel Beyene
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('SPAM_text.csv')
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
data['cat_label'] = pd.factorize(data['Category'])[0]

corpus = data['Message'].astype(str).tolist()
y = data['cat_label']

# Split the dataset into training and test sets
x_tr, x_te, y_tr, y_te = train_test_split(corpus, y, test_size=0.25, random_state=42)

# Apply TF-IDF vectorization to the text data
vectorizer = TfidfVectorizer()
x_tr = vectorizer.fit_transform(x_tr)
x_te = vectorizer.transform(x_te)

# Define the classifiers
classifiers = [
    ('SVM', SVC()),
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier())
]

best_model = None
best_metric = 0

# Evaluate each classifier and select the best one based on the chosen metric
for name, clf in classifiers:
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_te)

    acc = accuracy_score(y_pred, y_te)
    prc = precision_score(y_pred, y_te)
    rec = recall_score(y_pred, y_te)
    f1 = f1_score(y_pred, y_te)
    roc_auc = roc_auc_score(y_pred, y_te)

    print('Classifier:', name)
    print('Accuracy:', acc)
    print('Precision:', prc)
    print('Recall:', rec)
    print('F1-Score:', f1)
    print('ROC AUC:', roc_auc)
    print()

    # Update the best model based on the chosen metric
    if roc_auc > best_metric:
        best_model = clf
        best_metric = roc_auc

print('Best Model:', best_model)

# Save the best model
filename = 'Spam_Classification_best_model.pkl'
pickle.dump(best_model, open(filename, 'wb'))

# Save the vectorizer
vector_filename = 'Spam_Classification_vectorizer.pkl'
pickle.dump(vectorizer, open(vector_filename, 'wb'))