import streamlit as st
import numpy as np
import pickle  # For loading the trained model

# Load the trained KNN model
with open('iris_knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

# Streamlit app layout
st.title("Iris Flower Species Prediction")

st.write("""
This app predicts the species of an Iris flower based on the following features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
""")

# User input for flower features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button("Predict"):
    try:
        # Prepare input data
        input_data = np.asarray([sepal_length, sepal_width, petal_length, petal_width])
        input_data_reshaped = input_data.reshape(1, -1)
        
        # Make prediction
        prediction = knn_model.predict(input_data_reshaped)
        
        # Display the prediction result
        st.success(f"The predicted species is: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Error: {e}")
