from subword_tokenizer import SubwordTokenizer

tokenizer = SubwordTokenizer("tokenizer.json")

print(tokenizer.encode("states"))
print (tokenizer.decode([4, 58]))


#bütün metni tokenleştirelim----------------
with open("text.txt" , "r") as f:
    text = f.read()

print(text)

tokens = tokenizer.encode(text)
print(tokens)
#--------------------------------------------