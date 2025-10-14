#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 18:27:28 2025

@author: sinemertural
"""

"""
solve classification problem(sentiment analysis in NLP) with RNN (deep learning based language model)

duygu analizi -> bir cümlenin olumlu mu olumsuz mu olduğunu etiketleyeceğiz
restaurant yorumlari degerlendiricez

"""
#import libraries 
import numpy as np 
import pandas as pd

from gensim.models import Word2Vec #metin temsili 

#rnn için kütüphane
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# create database

data = {
    "text": [
        # ✅ Olumlu (Positive)
        "Yemekler harikaydı, çok lezzetliydi.",
        "Garsonlar çok güler yüzlüydü.",
        "Restoranın atmosferi çok hoştu, müzikler güzeldi.",
        "Tatlılar mükemmeldi, özellikle cheesecake.",
        "Fiyatlar makuldü ve ortam çok keyifliydi.",
        "Servis hızlıydı, siparişim hemen geldi.",
        "Yemeklerin sunumu çok şıktı.",
        "Personel çok ilgiliydi, kendimizi özel hissettik.",
        "Restoran tertemizdi, hijyene önem veriyorlar.",
        "Kahveleri harika, tam kıvamında.",
        "Menüde seçenek çok fazlaydı, her damak tadına uygun.",
        "Pizza inanılmazdı, hamuru tam kıvamında pişmişti.",
        "Restoran çok ferah, ışıklandırma harika.",
        "Garsonlar kibar ve profesyoneldi.",
        "Tatlılar gerçekten efsaneydi, porsiyonlar da büyük.",
        "Mekanın dekorasyonu modern ve sıcak.",
        "Yemekler sıcak ve taze geldi, mükemmel servis.",
        "Rezervasyon sistemi çok pratikti, zamanında oturduk.",
        "Aileyle gelmek için harika bir yer.",
        "Müzikler rahatsız edici değildi, ambiyans çok güzel.",
        "Kahvaltı tabağı dolu doluydu, çok doyurucuydu.",
        "Garsonlar menüyü çok güzel açıkladı, yardımcı oldular.",
        "Yemeklerin lezzeti beklentimin çok üzerindeydi.",
        "İçecekler soğuktu, servis hızlıydı.",
        "Tatlılar taze ve hafifti.",
        "Çocuklu aileler için uygun bir yerdi.",
        "Etler yumuşacık ve sulu pişmişti.",
        "Yemekler hem göze hem damağa hitap ediyor.",
        "Restoran sessizdi, rahat bir ortam vardı.",
        "Patates kızartması çıtır çıtırdı.",
        "Servis mükemmeldi, her şey zamanında geldi.",
        "Garsonlar çok nazikti, ilgililerdi.",
        "Çorba çok lezzetliydi, sıcak geldi.",
        "Mekanda wifi vardı, çok kullanışlı oldu.",
        "Tatlılar tam kararında şekerliydi.",
        "Restoranın konumu çok merkeziydi.",
        "Deniz manzaralı masalar çok güzeldi.",
        "Sunumlar özenle hazırlanmıştı.",
        "Fiyat performans açısından çok başarılıydı.",
        "Etler harika pişmişti, tam damak tadıma göreydi.",
        "Garsonlar sürekli ilgilendi, hiç beklemedik.",
        "Mekanın ışığı loş ve huzurluydu.",
        "Yemekler doyurucuydu, porsiyonlar büyük.",
        "Tatlıların yanında verilen kahve enfesti.",
        "Kebapları çok lezzetliydi, baharatı tam ayarında.",
        "Açık mutfak sistemi güven vericiydi.",
        "Mekanın temizliği harikaydı, lavabolar tertemizdi.",
        "Servis kusursuzdu, personel profesyoneldi.",
        "Müzik sesi idealdi, konuşmak kolaydı.",
        "Restorandan mutlu ayrıldım, herkese tavsiye ederim.",
        
        # ❌ Olumsuz (Negative)
        "Servis çok yavaştı, bir daha gelmem.",
        "Yemek soğuk geldi, hiç beğenmedim.",
        "Masalar çok kirliydi, hijyen hiç yoktu.",
        "Siparişim yanlış geldi, çok sinirlendim.",
        "Yemeklerin tadı berbattı, param boşa gitti.",
        "Garson ilgisizdi, defalarca çağırmam gerekti.",
        "Ortam çok gürültülüydü, rahat edemedim.",
        "Tatlı bayattı, yenilecek gibi değildi.",
        "Fiyatlar çok yüksekti, porsiyonlar küçük.",
        "Servis karışıktı, siparişlerimiz karıştırıldı.",
        "Restoran çok kalabalıktı, uzun süre bekledik.",
        "Kahve çok acıydı, içemedim.",
        "Sandalyeler rahatsızdı, oturmak işkenceydi.",
        "Garsonlar suratsızdı, ilgilenmediler.",
        "Menüdeki birçok ürün kalmamıştı.",
        "Yemek çok tuzluydu, yiyemedim.",
        "Patates kızartması yanmıştı.",
        "Pizzanın hamuru çiğ kalmıştı.",
        "Restoranda klima çalışmıyordu, çok sıcaktı.",
        "Tatlı istedik ama çok geç geldi.",
        "Müzik sesi çok yüksekti, konuşamadık.",
        "Yemeklerden kötü koku geliyordu.",
        "Fiyatlar kalitesine göre aşırı pahalıydı.",
        "Et sertti, çiğnemesi zordu.",
        "Garson siparişi yanlış getirdi.",
        "Servis süresi çok uzundu, 40 dakika bekledik.",
        "Tatlı aşırı şekerliydi, yiyemedik.",
        "Mekanda sinek vardı, rahatsız olduk.",
        "Masalar yapış yapıştı, temizlik kötüydü.",
        "Su bile parayla satılıyor, saçmalık.",
        "Restoranda park yeri yoktu.",
        "Çorbada taş gibi bir şey çıktı, şok oldum.",
        "Yemek porsiyonu çok küçüktü.",
        "Tatlıların tadı çok yapaydı.",
        "Garson kaba davrandı.",
        "Sandviç bayattı, zorla yedim.",
        "Yemekte saç buldum, iğrendim.",
        "Restoranın tuvaletleri çok kirliydi.",
        "Çatal bıçaklar lekeli geldi.",
        "Kola gazsızdı, bayattı.",
        "Fiyatlar menüyle uyuşmuyordu.",
        "Servis eksikti, su bile getirmediler.",
        "Rezervasyonum olmasına rağmen masa hazır değildi.",
        "Yemek geldiğinde zaten soğumuştu.",
        "Tatlılar bitmişti, başka bir şey istemek zorunda kaldık.",
        "Garson hesabı yanlış getirdi.",
        "Menüde fotoğraflar çok yanıltıcıydı.",
        "Kahve buz gibiydi, sıcak istemiştim.",
        "Restoran çok karanlıktı, rahatsız ediciydi.",
        "Yemekler baharatsızdı, tatsızdı."
    ],
    "label": [
        # 50 positive
        *["positive"]*50,
        # 50 negative
        *["negative"]*50
    ]
}


df = pd.DataFrame(data) # hangi yoruma hangi etiket geldiğini gösterir

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"]) #metinleri sayısal vektörlere çevirdi
word_index = tokenizer.word_index #kelimelere denk gelen sayı

#padding process 
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences,maxlen=maxlen)
print(X.shape)

#label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"]) #pozitifler 1 negatifler 0 oldu

#train test split
X_train , X_test , y_train , y_test=train_test_split(X, y,test_size=0.2, random_state=42)

#metin temsili : word embedding: word2vec

sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences,vector_size=50,window=5,min_count=1)

embedding_dim = 50
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word , i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]
 

#modelin build ,train ve test rnn modeli

#modelin inşa edilmesi
model = Sequential()

#embedding
model.add(Embedding(input_dim = len(word_index) + 1 , output_dim = embedding_dim, weights = [embedding_matrix], input_length = maxlen, trainable = False))
#rnn katmanı
model.add(SimpleRNN(50,return_sequences=False))
#output katmanı
model.add(Dense(1,activation="sigmoid"))

#compile model
model.compile(optimizer ="adam",loss = "binary_crossentropy",metrics=["accuracy"])
#train model
model.fit(X_train, y_train, epochs=10,batch_size =2 ,validation_data = (X_test,y_test))

#evaluate rnn modeli
test_loss,test_accuracy= model.evaluate(X_test,y_test)
print(f"Test loss : {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


#cumle sınıflandırma çalışması
def classify_sentence(sentence):
    
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq,maxlen =maxlen)
    prediction = model.predict(padded_seq)
    predicted_class = (prediction > 0.5).astype(int) 
    label = "positive" if predicted_class[0][0] ==  1 else "negative"
    
    return predicted_class

sentence = "Restaurant çok temizdi ve yemekler çok güzeldi"

result= classify_sentence(sentence)
print(f"Result: {result}")
    

















