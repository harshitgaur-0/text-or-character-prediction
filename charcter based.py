data = open("story.txt").read().lower()
import string
punc = list(string.punctuation)
data = list(data)

for char in punc:
    while True:
        try:
            data.remove(char)
        except:
            break
num = ["0","1","2","3","4","5","6","7","8","9","10"]
for char in num:
    while True:
        try:
            data.remove(char)
        except:
            break
data = "".join(data)
data = data.split()
raw_data = " ".join(data)

length = 10
seq = []

for i in range(length,len(raw_data)):
    s = raw_data[i-length : i+1]
    seq.append(s)

chars = sorted(list(set(raw_data)))
mapping = dict((c,i) for (i,c) in enumerate(chars))

token_seq = []
for text in seq:
    encoded_seq = []
    for char in text:
        encoded_seq.append(mapping[char])
    token_seq.append(encoded_seq)

import numpy as np
token_seq = np.array(token_seq)

train_x = token_seq[:,:-1]
train_y = token_seq[:,-1]
from keras.utils import to_categorical
train_y = to_categorical(train_y,num_classes=len(mapping))
from keras.models import Sequential
from keras.layers import Dense,Dropout,Bidirectional,Embedding,LSTM

import keras
class mycallbacks(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("accuracy")>0.98):
            print("98% accuracy reached")
            self.model.stop_trainig = True
callback1 = mycallbacks()

callback2 = keras.callbacks.EarlyStopping(monitor="loss",patience=5)


model = Sequential()

model.add(Embedding(len(mapping),16,input_length=10))
model.add(Bidirectional(LSTM(256,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(256)))
model.add(Dropout(0.2))
model.add(Dense(1024,activation="relu"))
model.add(Dense(len(mapping),activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(train_x,train_y,epochs=500,callbacks=[callback1,callback2])

import pickle
pickle.dump(model,open("character_based_model.sav","wb"))
pickle.dump(mapping,open("character_based_mapping.sav","wb"))
