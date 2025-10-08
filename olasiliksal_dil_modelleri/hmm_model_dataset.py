#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 09:57:59 2025

@author: sinemertural
"""

import nltk 
from nltk.tag import hmm
from nltk.corpus import conll2000

#gerekli veri paketini indir
nltk.download("conll2000")

#conll veri setini yükle
train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt")

#hmm training
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

#test 
test_sentence = "I am not going to park".split()
taggs = hmm_tagger.tag(test_sentence)