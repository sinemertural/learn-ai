#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:38:58 2025

@author: sinemertural
"""

import string
from textblob import TextBlob
from bs4 import BeautifulSoup
import re

text = "  <div> John  was walking  in   the park ,  when he  saw   a littel dog  near the  lake . it   was Barking loudly  and runing   around but nobody   seemed  to care . </div>  Then   he  decieded to  go closer  because  he   thought it might  be lost . on his way he met  an old  woman who  said she had  seen the same  dog earliar   in the  morning .They  both looked for it   together and  after   a  few   minutes the  dog ran  happly to   its  owner .  At that   momment everyone  in  the park  smiled ,   and claped their   hands !!"

# %% veri temizleme
#fazla boşluklardan kurtul
cleaned_text = " ".join(text.split())
print(cleaned_text)

#tüm kelimleri küçük harfle yaz
text_lower = cleaned_text.lower()
print(text_lower)

#punctuation (noktalama) kaldır
text_punctuation = text_lower.translate(str.maketrans("","",string.punctuation))
print(text_punctuation)

#yazım yanlışlarından kurtul
texy_typo = str(TextBlob(text_punctuation).correct())
print(texy_typo)

# html veya url etiketlerinin kaldırılması
text_unlabeled = BeautifulSoup(texy_typo, "html.parser").get_text()
print(text_unlabeled)

# ozel karakter kaldir
text_special_character = re.sub(r"[^A-Za-z0-9\s]", "", text_unlabeled)
print(text_special_character)

final_text = text_special_character
print(final_text)

# %% tokenizasyon

import nltk
nltk.download("punkt")

#kelimeleri tokenlara ayır
word_tokens = nltk.word_tokenize(final_text)
print(word_tokens)

#cümleleri tokenlara ayır
sentence_tokens = nltk.sent_tokenize(final_text)
print(sentence_tokens)

# %% stem ve lemma 
#stem
nltk.download("wordnet")
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

stems = [stemmer.stem(w) for w in word_tokens]
print("Stem results ->", stems)

#lemma
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmas = [lemmatizer.lemmatize(w , pos="v") for w in word_tokens]
print("Lemma results ->", lemmas)

# %% stop words
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

filtered_words = [w for w in word_tokens if w.lower() not in stop_words]
print(filtered_words)


# %% ödevin eksik yanları ve düzeltilmesi gereken kısımlar : metin ön işleme yaparken HTML → lower → fazla boşluk → (cümle tokenize) → noktalama → kelime tokenize → lemma → stopwords → stem yaptığında daha iyi analiz edebilirsin.






































