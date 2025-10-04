#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 19:38:47 2025

@author: sinemertural
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# veri seti yükle
df = pd.read_csv("sms_spam.csv")

#tf-idf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

#kelime kümesi 
feature_names = vectorizer.get_feature_names_out()
tfidf_skor = X.mean(axis = 0).A1 #ortalama tfidf değerleri

df_tfidf = pd.DataFrame({"word":feature_names , "tfidf_skor" : tfidf_skor}) #veri temizlemesi yapılması gerek bundan önce

#burada sıralama yaoarsak tfidfskor u en fazla öneme sahipp kelimeelri bulabiliriz
