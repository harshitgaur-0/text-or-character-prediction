data = open("story.txt").read()
data = data.split(".")

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000,oov_token="<oov>")
tokenizer.fit_on_texts(data)
num_words = len(tokenizer.word_index) + 1
train_tokenized_seq = tokenizer.texts_to_sequences(data)

while True:
    try:
        train_tokenized_seq.remove([])
    except:
        break

mapping = dict((i,c) for (c,i) in tokenizer.word_index.items())
max_len = max([len(arr) for arr in train_tokenized_seq])
import numpy as np
train_pad_token_seq = np.array(pad_sequences(train_tokenized_seq,maxlen=max_len,padding="pre",truncating="pre"))

from keras.utils import to_categorical
train_x = train_pad_token_seq[:,:-1]
train_y = train_pad_token_seq[:,-1]
train_y = to_categorical(train_y,num_classes=num_words)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,Bidirectional,Conv1D,LSTM

from keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor="loss",patience=10)
import keras
class mycallbacks(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get("accuracy")>0.98):
            print("accuracy is 98%")
            self.model.stop_training = True
callback_1 = mycallbacks()
model = Sequential()
model.add(Embedding(1000,16,input_length=max_len-1))

model.add(Bidirectional(LSTM(64)))

model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(num_words,activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
history = model.fit(train_x,train_y,epochs=500,callbacks=[callback,callback_1])

import pickle
pickle.dump(history,open("word_based_history.sav","wb"))
pickle.dump(model,open("word_based_,model.sav","wb"))
pickle.dump(mapping,open("word_based_mapping.sav","wb"))
pickle.dump(tokenizer,open("word_based_token.sav","wb"))