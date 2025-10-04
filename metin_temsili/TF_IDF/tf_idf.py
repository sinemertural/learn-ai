#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 19:11:06 2025

@author: sinemertural
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

#veri seti oluşturalım 
documents = [
    "Kedi çok tatlı bir hayvandır",
    "Kedi ve köpekler çok tatlı hayvanlardır",
    "Arılar bal üretirler"
    ]

tdifd_vectorizer = TfidfVectorizer()

#metinleri sayısal değerlere çevirelim
X = tdifd_vectorizer.fit_transform(documents)

#kelime kümesini alalım 
feature_names = tdifd_vectorizer.get_feature_names_out()

print("TF-IDF Vektör temsilleri : ")
vektor_temsili = X.toarray()
print(vektor_temsili)

df_tfidf = pd.DataFrame(vektor_temsili,columns = feature_names)

kedi_tfidf = df_tfidf["kedi"]
kedi_mean_tfidf = np.mean(kedi_tfidf)