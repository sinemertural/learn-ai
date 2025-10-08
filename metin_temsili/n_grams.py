#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 15:34:51 2025

@author: sinemertural
"""

from sklearn.feature_extraction.text import CountVectorizer

# ornek metin 
document = [
    "Bu bir örnek metindir."
    "Bu örnek metin doğal dil işlemeyi gösterir."
    ]

# uni gram bi gram ve tri gram oluşturmak içn countvectorizer kullanacağız

vectorizer_unigram = CountVectorizer(ngram_range=(1,1))
vectorizer_bigram = CountVectorizer(ngram_range=(2,2))
vectorizer_trigram = CountVectorizer(ngram_range=(3,3))

#unigram için fit transform 
X_unigram = vectorizer_unigram.fit_transform(document)
unigram_features = vectorizer_unigram.get_feature_names_out()

#bigram için fit transform 
X_bigram = vectorizer_bigram.fit_transform(document)
bigram_features = vectorizer_bigram.get_feature_names_out()

#unigram için fit transform 
X_trigram = vectorizer_trigram.fit_transform(document)
trigram_features = vectorizer_trigram.get_feature_names_out()