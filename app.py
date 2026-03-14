import streamlit as st
import tensorflow as tf
import pickle
import string
import re
import nltk

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

model = tf.keras.models.load_model("model.h5")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

max_len = 100

def clean_text(text):

    text = text.lower()
    text = re.sub(r'\d+','',text)

    text = text.translate(str.maketrans('','',string.punctuation))

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

def predict_email(email_text):

    email_text = clean_text(email_text)

    seq = tokenizer.texts_to_sequences([email_text])
    seq = pad_sequences(seq,maxlen=max_len)

    prediction = model.predict(seq)[0][0]

    if prediction > 0.5:
        return "🚨 Spam Email", prediction
    else:
        return "✅ Genuine Email", prediction

st.title("📧 AI Spam Email Detector")

st.write("Enter an email message below to check if it is Spam or Genuine.")

email_text = st.text_area("Email Content")

if st.button("Check Email"):

    if email_text.strip() == "":
        st.warning("Please enter email text")
    else:

        result,score = predict_email(email_text)

        st.subheader(result)
        st.write("Confidence Score:", round(float(score),3)) 