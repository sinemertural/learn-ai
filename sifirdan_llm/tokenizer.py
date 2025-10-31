#en ilkel tokenize işlemi ama hiçbir model elinde sonsuz tokenize işlemi sonucu oluşan kelime yutamaz bu fonksiyon bu yüzden iyileşmeli
text = "The capital of France is"

def tokenize(text):
  print(text.split())

tokenize(text)

#----------------------------------------
text1 = "The cat chased the dog"
text2 = "The dog chased the cat"
text = "The capital of France is"

vocab = {
  "The" : 0,
  "cat" : 1,
  "dog" : 2,
  "chased" : 3,
  "capital" : 4,
  "of" : 5,
  "France" : 6,
  "is" : 7 ,
  "<unk>" : 8
}

def tokenizer2(text):
  """
  Verilen metni kelimelere ayırır ve her kelimeyi vocab'daki ID'sine dönüştürür.
  """
  parts = text.split()
  ids = []
  for part in parts:
    if part in vocab:
      value = vocab[part]
    else:
      value = vocab["<unk>"]
    ids.append(value)
  
  return ids
  
token_ids1 = tokenizer2(text) 
token_ids2 = tokenizer2("How are you") #bilinmeyen
print(token_ids1)
print(token_ids2)

#--------------------------------------------------------
# şimdi detokenizer yapacağım yani id verip kelimeyi bulmasını isteyeceğim
reverse_vocab = {
  id : part for part, id in vocab.items()
}
reverse_vocab


def detokenizer(ids):
    text = ""
    for id in ids:
        part = reverse_vocab[id]
        text += part + " "
    return text.strip()

detokenizer = detokenizer(token_ids2)

"""
Büyük dil modelleri bu şekilde sözlük ve id mantığı ile çalışır. Örneğin : gemma , gpt-4o
"""

#--------------------------------------------------------
# şimdi büyük dil modellerinde hazır olan tokenizer kullanalim

import tiktoken 
enc = tiktoken.get_encoding("gpt2")
encode=enc.encode(text1)  # encode id ye çevirir

decode = enc.decode(encode) #decode aldığı id yi metne çevirir


#--------------------------------------------------------
""" artık tokenizer.json dosyası yani benim vacob ım gibi düşün bunu çeken 
bir kod yazacağım ve artık cümlelerimi bu tokenizer.json dosyasından çekeceğim 
"""
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it")

trasformers_processor_encode =processor.tokenizer.encode(text1)
trasformers_processor_decode =processor.tokenizer.decode(trasformers_processor_encode) # <bos>The cat chased the dog
transformers_processor_decode_str = processor.tokenizer.decode(trasformers_processor_encode)[5:]






















