import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Function to build and load the model
@st.cache_resource
def load_trained_model():
    # Build the model architecture
    conv_base = Xception(weights=None, include_top=False, input_shape=(299, 299, 3))

    model = Sequential([
        Input(shape=(299, 299, 3)),
        conv_base,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Load the saved model weights
    model.load_weights('C_CT_S.keras')
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Function to preprocess the uploaded image
def preprocess_image(img):
    # Convert image to RGB if it has an alpha channel (i.e., 4 channels)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((299, 299))  # Resize image to the required size
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch size
    img_array = img_array / 255.0  # Normalize the image
    return img_array


# Load the trained model
model = load_trained_model()

# Streamlit app interface
st.title("Lung Cancer Classification Web App")

st.write("Upload a lung CT scan image to predict its classification")

# File uploader to upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    st.write("Classifying...")
    
    # Preprocess the image and make prediction
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)[0][0]  # Get the prediction score

    # Convert prediction score to binary class
    if prediction <  0.5:
        st.write(f"Prediction: **Malignant** (Confidence: {prediction:.2f})")
    else:
        st.write(f"Prediction: **Benign** (Confidence: {1 - prediction:.2f})")
