import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Streamlit app for Spam Email Detection
st.title("Spam Email Detection App")
st.write("""
This app classifies email messages as Spam or Ham based on their content.
""")

# Load pre-trained model and vectorizer
@st.cache_resource
def load_model():
    # Assuming model and vectorizer are saved as pickle files
    with open('spam_classifier_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model()

# Input email text
input_email = st.text_area("Enter the email content:", value="")

# Predict button
if st.button("Classify"):
    try:
        if input_email.strip() == "":
            st.warning("Please enter some email content to classify.")
        else:
            # Transform input email
            input_features = vectorizer.transform([input_email])
            
            # Predict
            prediction = model.predict(input_features)
            label = "Ham" if prediction[0] == 1 else "Spam"
            
            # Display result
            st.success(f"The email is classified as: *{label}*")
    except Exception as e:
        st.error(f"Error: {e}")