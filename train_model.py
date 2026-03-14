import pandas as pd
import numpy as np
import string
import pickle
import re
import nltk

from nltk.corpus import stopwords
nltk.download('stopwords')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

print("Loading dataset...")

data = pd.read_csv("train.csv")

data = data[['text','label']]

# Balance dataset
ham = data[data['label']=="ham"]
spam = data[data['label']=="spam"]

ham = ham.sample(len(spam), random_state=42)
data = pd.concat([ham,spam])

print("Dataset balanced")

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = text.lower()
    text = re.sub(r'\d+','',text)

    text = text.translate(str.maketrans('','',string.punctuation))

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

print("Cleaning text...")

data['text'] = data['text'].apply(clean_text)

X_train,X_test,y_train,y_test = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2,
    random_state=42
)

y_train = (y_train=="spam").astype(int)
y_test = (y_test=="spam").astype(int)

print("Tokenizing text...")

tokenizer = Tokenizer(num_words=5000,oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

train_seq = tokenizer.texts_to_sequences(X_train)
test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 100

train_seq = pad_sequences(train_seq,maxlen=max_len)
test_seq = pad_sequences(test_seq,maxlen=max_len)

print("Building model...")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000,64,input_length=max_len),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

print("Training model...")

model.fit(
    train_seq,
    y_train,
    epochs=5,
    validation_data=(test_seq,y_test),
    batch_size=32
)

print("Saving model...")

model.save("model.h5")

with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

print("Model saved successfully!") 