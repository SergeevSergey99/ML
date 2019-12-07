import os

import numpy as np

# Neural Network
from keras.models import load_model
# Text Vectorizing
from keras.preprocessing.text import Tokenizer


model = load_model('model.h5')

tokenizer = Tokenizer(num_words=10000, 
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                      lower=True, 
                      split=' ', 
                      char_level=False)

def isNotGood(str):
    v = model.predict(tokenizer.texts_to_matrix([str]))
    if v > 0.7:
        print("Ругательства")
    if v <= 0.7 and v > 0.3:
        print("Средне")
    if v <= 0.3:
        print("Конструктивная речь")
        
isNotGood("JavaScript")