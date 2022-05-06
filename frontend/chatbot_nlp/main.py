# importing the libraries
import json
import pickle
import random
import string
import urllib
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow.keras.layers import (LSTM, Dense, Embedding, Flatten,
                                     GlobalMaxPooling1D, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# importing the dataset
#with open('content.json') as content:
#   data1 = json.load(content)

#url = "https://jsonkeeper.com/b/OBM5"
#response = urllib.request.urlopen(url)
#data1 = json.loads(response.read())
#importing the dataset
dir_name = os.path.abspath(os.path.dirname('main.py'))
location = os.path.join(dir_name, 'content.json')
with open(location) as content:
  data1 = json.load(content)
# getting all the data to lists
tags = []
inputs = []
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['input']:
        inputs.append(lines)
        tags.append(intent['tag'])
# converting to dataframe
data = pd.DataFrame({"inputs": inputs,
                     "tags": tags})

model = tf.keras.models.load_model('my_model')
tokenizer = Tokenizer(num_words=2000)

# removing punctuations
data['inputs'] = data['inputs'].apply(
    lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))
# encoding the outputs
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])
# tokenize the data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
# apply padding
x_train = pad_sequences(train)
input_shape = x_train.shape[1]

url = 'https://drive.google.com/file/d/1-zyO_hWGYLruIuskGQX-vpwU8LgzEmhO/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)

list = df['Symptom'].tolist()


def disease_prediction(symptoms):
    url = 'https://drive.google.com/file/d/1-zyO_hWGYLruIuskGQX-vpwU8LgzEmhO/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df1 = pd.read_csv(path)
    columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
               'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9',
               'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14',
               'Symptom_15', 'Symptom_16', 'Symptom_17']
    df = pd.DataFrame(columns=columns)
    data = [0] * 17
    input_list = symptoms
    for element, i in zip(input_list, range(len(input_list))):
        data[i] = element
    df.loc[len(df)] = data
    df.isna().sum()
    df.isnull().sum()

    cols = df.columns
    data = df[cols].values.flatten()

    s = pd.Series(data)
    #s = s.str.strip()
    s = s.values.reshape(df.shape)

    df = pd.DataFrame(s, columns=df.columns)

    df = df.fillna(0)
    vals = df.values
    symptoms = df1['Symptom'].unique()

    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom']
                                        == symptoms[i]]['weight'].values[0]

    d = pd.DataFrame(vals, columns=cols)

    d = d.replace('dischromic _patches', 0)
    d = d.replace('spotting_ urination', 0)
    df = d.replace('foul_smell_of urine', 0)

    filename = 'disease_prediction.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    preds = loaded_model.predict(df)

    return preds


def disease_analysis(pred):
    url = 'https://drive.google.com/file/d/1vHqJum4ea7LP6bPgiXJGxW4-J2HgyM3R/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df = pd.read_csv(path)

    disease_description = df[df['Disease'] == pred]['Description']
    st.subheader('Disease description:')
    st.write(disease_description.values[0])

    url = 'https://drive.google.com/file/d/1Ne1Be5y4im4wk7FRmBukrDp2iDwnPH42/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df1 = pd.read_csv(path)

    disease_precaution = df1[df1['Disease'] == pred]
    disease_precaution.dropna(axis=1,inplace=True)
    disease_precaution = disease_precaution.iloc[0].to_numpy()[1:]
    st.subheader('Disease precaution:')
    for element in disease_precaution:
        st.write(element)


def symptom_extractor(options):
    pred = disease_prediction(options)
    st.subheader('Disease predicted:', pred[0])
    st.write(pred[0])
    disease_analysis(pred[0])


input_message = st.text_input('input', '')


def func():
    texts_p = []
    #predition_input = input('You : ')
    predition_input = input_message
    # removing punctuation and converting to lowercase
    predition_input_1 = [letters.lower(
    ) for letters in predition_input if letters not in string.punctuation]
    predition_input_1 = ''.join(predition_input_1)
    texts_p.append(predition_input_1)
    #tokenizing and padding
    predition_input_1 = tokenizer.texts_to_sequences(texts_p)
    predition_input_1 = np.array(predition_input_1).reshape(-1)
    predition_input_1 = pad_sequences([predition_input_1], input_shape)
    # getting output from model
    output = model.predict(predition_input_1)
    output = output.argmax()
    # finding the right tag and predicting
    response_tag = le.inverse_transform([output])[0]
    #print("You : ", predition_input)
    #print("Sunbot : ",random.choice(responses[response_tag]))
    if response_tag == "tiggerwords":
        with st.form("my_form"):
            options = st.multiselect(
                'What are your symptoms',
                list,
                ['itching'])
            submitted = st.form_submit_button("submit")
            if submitted:
                symptom_extractor(options)

    output_message = random.choice(responses[response_tag])
    st.markdown(output_message)


button1 = st.button('enter')
if st.session_state.get('button') != True:
    st.session_state['button'] = button1

if st.session_state['button'] == True:
    func()
