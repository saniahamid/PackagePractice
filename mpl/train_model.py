# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 21:29:20 2021

@author: hamid
"""
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

from dataset import Dataset

dset = Dataset()
X_train, y_train = dset.get_train_set(limit=10000)
X_test, y_test = dset.get_test_set(limit=10000)


tfidf = TfidfVectorizer(X_train, max_features=10000)


X_train = tfidf.fit_transform(X_train)

with open('tfidf.pickle', 'wb') as f:
    pickle.dump(tfidf, f)


clf = MultinomialNB()
clf.fit(X_train, y_train)


with open('model.pickle', 'wb') as f:
    pickle.dump(clf, f)


X_test = tfidf.transform(X_test)
y_pred = clf.predict(X_test)


print(classification_report(y_test, y_pred))
