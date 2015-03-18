"""
Beating the benchmark 
Otto Group product classification challenge @ Kaggle

__author__ : Abhishek Thakur
"""

import pandas as pd
import scipy as sp
import numpy as np
from __future__ import division
from sklearn.cross_validation import train_test_split
from sklearn import ensemble, feature_extraction, preprocessing

# multiclass loss
def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

# import data
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
sample = pd.read_csv('Data/sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# set up datasets for cross eval
x_train, x_test, y_train, y_test = train_test_split(train, labels)

# train a random forest classifier
clf = ensemble.RandomForestClassifier(n_estimators = 200, verbose = 1)
clf.fit(x_train, y_train)

# predict on test set
preds = clf.predict_proba(x_test)

# ----------------------  create submission file  -----------------------------
#preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
#preds.to_csv('benchmark.csv', index_label='id')

# ----------------------  cross eval  -----------------------------------------

    scores = []
    for index in range(0, len(pred)):
        result = llfun(y_test[index], preds[index])
        scores.append(result)

    print(sum(scores) / len(scores)) 