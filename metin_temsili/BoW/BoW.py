#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 15:24:49 2025

@author: sinemertural
"""

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Kedi evde" , 
    "Kedi bahçede"
    ]

vectorizer = CountVectorizer()

#metinlerden sayısal vektöre çeviriyorum
X = vectorizer.fit_transform(documents)

#kelime kümesi["kedi" , "evde" , "bahçede"] 
print("Kelime Kümesi: " ,vectorizer.get_feature_names_out())

# vektör temsili 
print("Vektöre Temsili: " , X.toarray())