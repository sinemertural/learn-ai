#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:04:09 2025

@author: sinemertural
"""

import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

# stop word liste yukle
stop_words = set(stopwords.words("english"))

#ornek metin
text = "This is an example of removing stop words from a text document."

filtered_words = [word for word in text.split() if word.lower() not in stop_words]

print(filtered_words)

# ornek metin türkçe
stop_words_tr = set(stopwords.words("turkish"))
text2 = "merhaba dünya ve bu güzel insanlar"

filtered_words_tr = [word for word in text2.split() if word.lower() not in stop_words_tr]

print(filtered_words_tr)

# %%

turkish_stopwords = set(["ve" , "bir" , "bu", "ile"])

#ornek metin 
text = "Bu bir örnek metin ve stop words'leri temizlemek için kullanılıyor."
filtered_words_tr = [word for word in text.split() if word.lower() not in turkish_stopwords]
print(filtered_words_tr)


