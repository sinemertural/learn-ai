#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 17:57:29 2025

@author: sinemertural
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

df = pd.read_csv("IMDB Dataset.csv")

df2 = df.head(100) #ilk 100 veriyi alalım

#metin verilerini alalim 
documents = df["review"] #metin
label = df["sentiment"] # positive or negative

# öncelikle metin ön işleme yaparak metni temizleyelim 
def clean_text(text):
    
    #kucuk harf cevrimi
    text = text.lower()
    
    #rakam temizle
    text = re.sub(r"\d+", "", text)
    
    #ozel karakterleri temizle
    text = re.sub(r"[^\w\s]", "", text)
    
    #kisa kelimeleri temizle (tek harfli kelimeler -> a ) olabilir
    text = " ".join([word for word in text.split() if len(word) > 2]) 

    
    return text
    

#metinleri temizle 
cleaned_documents = [clean_text(doc)for doc in documents]

#BoW
vectorizer = CountVectorizer()

#temizlenmiş metinlerden sayısal vektör
X = vectorizer.fit_transform(cleaned_documents[:100])

#kelime kümesi
feature_names = vectorizer.get_feature_names_out()

#vektör temsili 
print("Vektör temsili")
vektor_temsili_2 = X.toarray()[:2]
print(vektor_temsili_2)

#vektör temsili dataframe
df_bow = pd.DataFrame(X.toarray(), columns=feature_names)

#kelime frekansı 
word_counts = X.sum(axis=0).A1
word_frequency = dict(zip(feature_names, word_counts))

# ilk 5 kelime
most_common_words = Counter(word_frequency).most_common(5)

