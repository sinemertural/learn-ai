# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %%metinlerdeki fazla boşluğu temizle
text = "Hello,     World!        2025"  #"Hello, World! 2025"
text.split()
cleaned_text1=" ".join(text.split())
print(cleaned_text1)

# %%büyükten küçüğe harf çevrimi
text= "Hello, World! 2025"
cleaned_text2 = text.lower()
print(cleaned_text2)

# %%noktalama işaretlerini kaldır
import string

text= "Hello, World! 2025"
cleaned_text3 = text.translate(str.maketrans("","",string.punctuation))
print(cleaned_text3)

# %% yazım hatalarını duzelt
from textblob import TextBlob

text= "Hellıo, Wirld! 2025"
cleaned_text5 = str(TextBlob(text).correct())
print(cleaned_text5)

# %% html veya url etiketlerinin kaldırılması
from bs4 import BeautifulSoup

html_text= "<div>Hello, World! 2025</div>"
cleaned_text6 = BeautifulSoup(html_text, "html.parser").get_text()
print(cleaned_text6)

# %% ozel karakter kaldir
import re

text= "Hello, World! 2025"
cleaned_text7 = re.sub(r"[^A-Za-z0-9\s]", "", text)
print(cleaned_text7)



