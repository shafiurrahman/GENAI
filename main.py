import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

#load the word_index from dataset imdb
word_index=imdb.get_word_index()
reverse_word_index={value:key for (key,value) in word_index.items()}

#load the pre-trained modelswith RELU ACTIVATION
model=load_model('F:\GenAI\simpleRNN\simplernn_imdb.h5')


#step2 helper function
#function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])

#function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#step 3: prediction function
### Prediction  function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

#streamlit
import streamlit as st
st.title(" IMDB MOVIE SENTIMENT ANALYSIS:prediction using simple RNN")
st.write("enter a movie reveiw so that it can be classeified as positive or negative:")

#USER INPUT
user_input=st.text_area("movie review")
if st.button("classify"):
    pre_processed_input=preprocess_text(user_input)

    #make prediction
    prediction=model.predict(pre_processed_input)
    sentiment= 'positive' if prediction[0][0]>0.5 else 'Negative'

    #display the result
    st.write(f'sentiment:{sentiment}')
    st.write(f'prediction score:{prediction[0][0]}')
else:
    st.write('please enter movie review')


