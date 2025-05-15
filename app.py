import streamlit as st
import joblib
import string
import os

# Load model and vectorizer
try:
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    st.error("❌ Model files not found! Make sure 'fake_news_model.pkl' and 'vectorizer.pkl' are in the same folder as this app.")
    st.stop()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Paste any news article or text below, and this tool will predict whether it's **real** or **fake**.")

# Text input
text_input = st.text_area("Enter News Text Here:")

# Predict button
if st.button("Check"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some news text before checking.")
    else:
        cleaned = preprocess(text_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("🚨 This appears to be FAKE news.")
        else:
            st.success("✅ This appears to be REAL news.")
import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Test a real news example
text = "NASA’s Artemis I mission successfully completes its journey around the Moon."
vectorized_text = vectorizer.transform([text])
prediction = model.predict(vectorized_text)

print("📰 Prediction:", "FAKE" if prediction[0] == 1 else "REAL")
