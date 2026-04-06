from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load models
model_cat = joblib.load("../model/category_model.pkl")
model_pri = joblib.load("../model/priority_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    return " ".join([w for w in words if w not in stop_words])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    
    clean = preprocess(data)
    vec = vectorizer.transform([clean])
    
    category = model_cat.predict(vec)[0]
    priority = model_pri.predict(vec)[0]
    
    return jsonify({
        "category": category,
        "priority": priority
    })

if __name__ == "__main__":
    app.run(debug=True)