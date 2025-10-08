#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 09:49:14 2025

@author: sinemertural
"""

import nltk
from nltk.tag import hmm

#ornek veri seti
train_data = [
    [("I" , "PRP"),("am" , "VBP"),("a" , "DT"),("student" , "NN")],
    [("You" , "PRP"),("are" , "VBP"),("a" , "DT"),("teacher" , "NN")], 
    ]

#hmm training
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# yeni cümle
test_sentence = "I am a teacher".split()
tags = hmm_tagger.tag(test_sentence)

print("Etiketli cğmle: ",tags)

#texti aslında etiketlemiş ve eğitmiş olduk 
