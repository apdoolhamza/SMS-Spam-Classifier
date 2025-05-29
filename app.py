from flask import Flask,render_template, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load model and vectorizer
try:
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')
except FileNotFoundError:
    raise Exception("Model or TF-IDF vectorizer file not found. Make sure 'model.pkl' and 'tfidf.pkl' exist.")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)           # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)       # Remove punctuation
    return text.strip()

@app.route("/")
def home():
    return render_template("index.html")

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    email_text = data.get('email', '')

    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400

    cleaned = preprocess_text(email_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    result = "Spam" if prediction == 'spam' else "Not Spam"

    return jsonify({
        'email': email_text,
        'prediction': result
    })

if __name__ == '__main__':
    app.run(debug=True)