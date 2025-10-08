#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 10:28:19 2025

@author: sinemertural
"""

from nltk.classify import MaxentClassifier

#egitim seti
train_data = [
    ({"Love" : True , "amazing": True} , "positive"),
    ({"hate" : True , "terrible": True} ,"negative"),
    ({"happy" : True , "joy": True }, "positive"),
    ({"sad": True , "depressed": True }, "negative")]

#max entropy training
classifier=MaxentClassifier.train(train_data)


#test 
test_sentence = "I love this amazing maovie"
features = {word:(word in test_sentence.lower().split()) for word in ["Love","amazing" , "hate","terrible","happy","joy","sad","depressed"]}
label= classifier.classify(features)
print(label)