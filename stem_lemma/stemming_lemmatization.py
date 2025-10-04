#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 13:34:26 2025

@author: sinemertural
"""

import nltk
nltk.download("wordnet")

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

#ornek kelimeler
words = ["running", "runner" , "ran" , "runs", "better" , "go" , "went"]


stems = [stemmer.stem(w) for w in words]

print("Stem results ->", stems)

# %% lemma

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

#ornek kelimeler
words = ["running", "runner" , "ran" , "runs", "better" , "go" , "went"]

lemmas = [lemmatizer.lemmatize(w, pos="v") for w in words] # v -> verb , n -> noun

print("Lemma results ->" ,lemmas)