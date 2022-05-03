import tensorflow as tf
#importing the libraries
import json
import string
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM , Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.models import Model


#importing the dataset
with open('content.json') as content:
  data1 = json.load(content)


#getting all the data to lists
tags = []
inputs = []
responses={}
for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])

#converting to dataframe
data = pd.DataFrame({"inputs":inputs,
                     "tags":tags})

model = tf.keras.models.load_model('my_model')
tokenizer = Tokenizer(num_words=2000)
input_shape = 7


#removing punctuations
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

#encoding the outputs
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])
def symptom_extractor():
    print("tell me your symptoms")

while True:
  texts_p = []
  predition_input = input('You : ')
  #removing punctuation and converting to lowercase
  predition_input_1 = [letters.lower() for letters in  predition_input if letters not in string.punctuation]
  predition_input_1 = ''.join(predition_input_1)
  texts_p.append(predition_input_1)
  #tokenizing and padding
  predition_input_1 = tokenizer.texts_to_sequences(texts_p)
  predition_input_1 = np.array(predition_input_1).reshape(-1)
  predition_input_1 = pad_sequences([predition_input_1],input_shape)
  #getting output from model
  output = model.predict(predition_input_1)
  output = output.argmax()
  #finding the right tag and predicting
  response_tag = le.inverse_transform([output])[0]
  print("You : ", predition_input)
  print("Going Merry : ",random.choice(responses[response_tag]))
  if response_tag == "goodbye":
    break
  elif response_tag == "tiggerwords":
    symptom_extractor()