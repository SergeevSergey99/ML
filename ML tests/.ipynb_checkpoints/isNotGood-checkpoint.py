import os
import numpy as np
# Neural Network
from keras.models import load_model
# Text Vectorizing
from keras.preprocessing.text import Tokenizer
import pickle
# loading

model = load_model('toxic_model75-20.h5')
with open('tokenizer75-20.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def isNotGood(str):
    v = model.predict(tokenizer.texts_to_matrix([str]))
    return v