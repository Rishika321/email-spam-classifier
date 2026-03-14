# Email Spam Classifier

A machine learning web application that detects whether an email message is **Spam** or **Not Spam** using an LSTM-based deep learning model.

This project uses Natural Language Processing (NLP) techniques to preprocess email text and classify it using a neural network built with **TensorFlow** and **Keras**. The application interface is built using **Streamlit**.

---

## 🚀 Features

* Email text preprocessing
* Stopword removal using NLTK
* Tokenization and sequence padding
* LSTM deep learning model
* Real-time spam prediction
* Interactive web interface

---

## 🧠 Technologies Used

* Python
* TensorFlow
* Keras
* Streamlit
* Scikit-learn
* NLTK
* Pandas
* NumPy

---

## 📂 Project Structure

```
email-spam-classifier
│
├── app.py               # Streamlit web application
├── train_model.py       # Script to train the LSTM model
├── train.csv            # Dataset used for training
├── model.h5             # Trained model file
├── tokenizer.pkl        # Saved tokenizer
├── requirements.txt     # Project dependencies
├── .gitignore           # Files ignored by Git
└── README.md            # Project documentation
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Rishika321/email-spam-classifier.git
```

Navigate to the project directory:

```
cd email-spam-classifier
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the Streamlit app:

```
streamlit run app.py
```

The application will start locally and open in your browser.

---

## 📊 How It Works

1. Email text is cleaned and preprocessed.
2. Stopwords are removed using NLTK.
3. Text is converted into numerical sequences using a tokenizer.
4. Sequences are padded to a fixed length.
5. The trained LSTM model predicts whether the message is spam or not.


