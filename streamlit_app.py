import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.write("Files in root:", os.listdir(BASE_DIR))

model_path = os.path.join(BASE_DIR, "model")

if os.path.exists(model_path):
    st.write("Files in model folder:", os.listdir(model_path))
else:
    st.write(" model folder NOT FOUND")
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_cat = joblib.load(os.path.join(BASE_DIR, "model", "category_model.pkl"))
model_pri = joblib.load(os.path.join(BASE_DIR, "model", "priority_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    return " ".join([w for w in words if w not in stop_words])

st.title("🎫 Customer Support Ticket Classifier")

user_input = st.text_area("Enter your issue:")

if st.button("Predict"):
    clean = preprocess(user_input)
    vec = vectorizer.transform([clean])

    category = model_cat.predict(vec)[0]
    priority = model_pri.predict(vec)[0]

    st.success(f"Category: {category}")
    st.success(f"Priority: {priority}")
