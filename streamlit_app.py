# streamlit_app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('exported_model.zip')

# Mapping for class labels
class_labels = {
    0: "Battery",
    1: "Glass",
    2: "Metal",
    3: "Organic",
    4: "Paper",
    5: "Plastic"
}

# Function to preprocess image for model prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Streamlit app
st.title("Waste Classification App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Make a prediction using the loaded model
    img_array = preprocess_image(uploaded_file)
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Display prediction result
    st.subheader("Prediction Result:")
    st.write(f"Predicted Class: {class_labels[predicted_class_index]}")
    st.write(f"Probability: {predictions[0][predicted_class_index] * 100:.2f}%")
