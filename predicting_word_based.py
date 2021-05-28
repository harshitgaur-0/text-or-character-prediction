import pickle
mapping = pickle.load(open("word_based_mapping.sav","rb"))
model = pickle.load(open("word_based_,model.sav","rb"))
tokenizer = pickle.load(open("word_based_token.sav","rb"))
original_text = "Hello my name is Mr X"
text = original_text.split()
from keras.preprocessing.sequence import pad_sequences
token_text = tokenizer.texts_to_sequences(text)
pad_token_text = pad_sequences(token_text,maxlen=86)
predicted = model.predict_classes(pad_token_text)
next_word = ""
for i in range(len(predicted)):
    for (char,index) in tokenizer.word_index.items():
        if index == predicted[i]:
            next_word = char
            break
    print(original_text + " " + next_word)