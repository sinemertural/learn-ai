#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 08:36:52 2025

@author: sinemertural
"""

from collections import Counter
import nltk 
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

#ornek veri seti 
corpus = [
    "I love apple.",
    "I love  you",
    "I love programming",
    "You love me",
    "She loves me",
    "They love you",
    "I love you an you love me",
    ]

#tokenize
tokens = [word_tokenize(sentence.lower()) for sentence in corpus ]

#n-gram -> n:2

bigrams = []
for token_list in tokens:
    bigrams.extend(ngrams(token_list, 2))
    
    
#bigramlardaki frekansları sayalım (counter)
bigrams_freq=Counter(bigrams)

#n -> 3 olsun 
trigrams = []
for token_list in tokens:
    trigrams.extend(ngrams(token_list, 3))
    
#trigramlardaki frekansları sayalım (counter)
trigrams_freq=Counter(trigrams)

# "I love" bigramından sonra you veya apple gelme olasılıklarını hesapla
bigram = ("i" , "love")
prob_you = trigrams_freq[("i","love","you")]/bigrams_freq[bigram]
prob_apple = trigrams_freq[("i","love","apple")]/bigrams_freq[bigram]

print("you kelimesinin olma olasılığı:",prob_you)
print("apple kelimesinin olma olasılığı:",prob_apple)