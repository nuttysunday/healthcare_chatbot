import streamlit as st
from streamlit_chat import message
import requests
import random

#row1
col1, col2  = st.columns([1000, 1])
#row2
col3, col4  = st.columns([1000, 1])

response = ['wassup','bitch','fuck off']

with col1:
    message("hello! I am your personal health care chatbot!") 

with col3:   
    user_input = st.text_input("Enter your message...", "")

with col1:
    message(user_input, is_user=True)

if user_input != "":
    with col1:
        message(random.choice(response)) 