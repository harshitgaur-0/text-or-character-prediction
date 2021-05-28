import pickle
model = pickle.load(open("character_based_model.sav","rb"))
mapping = pickle.load(open("character_based_mapping.sav","rb"))

text = "that day w"
seq = []
for char in text:
    seq.append(mapping[char])
import numpy as np
seq = np.array([seq])
predicted_class = model.predict_classes(seq)
next_char = ""
for (c,i) in mapping.items():
    if i ==predicted_class:
        next_char=c
text+=next_char
print(text)


