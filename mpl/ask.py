# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 21:58:52 2021

@author: hamid
"""
import pickle

tfidf = None
clf = None

with open('tfdif.pickle', 'rb') as f:
    tfdif = pickle.load(f)


with open('model.pickle', 'rb') as f:
    clf = pickle.load(f)

x = input("Please enter your phrase: ")
y = clf.predict_proba(tfidf.transform([x]))
print(y[0])
