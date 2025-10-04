#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 19:02:56 2025

@author: sinemertural
"""

import nltk

nltk.download("punkt")

text = "Hello, World! How are you?"

#kelimeleri tokenlara ayır
word_tokens = nltk.word_tokenize(text)
print(word_tokens)

#cumle tokenization
sentence_tokens = nltk.sent_tokenize(text)
print(sentence_tokens)
