import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the Xception model with your architecture
conv_base = Xception(weights=None, include_top=False, input_shape=(299, 299, 3))

# Load the weights from the local file
conv_base.load_weights('xception_weights_tf_dim_ordering_tf_kernels.h5')

# Load the model architecture again without loading weights
model = Sequential([
    conv_base,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Adjust output for your 4 classes
])
print("done")
# Load weights by name
model.load_weights('lung_cancer_xception.weights.h5', by_name=True)


# Load any additional model weights if necessary
model.load_weights('lung_cancer_xception.weights.h5')  # If you have additional weights

# Streamlit web interface
st.title("Lung Cancer Classification")

# File uploader to load an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)
    class_names = ['Adenocarcinoma', 'Squamous Cell Carcinoma', 'Normal', 'Large Cell Carcinoma']
    predicted_class = class_names[np.argmax(predictions)]
    
    # Display results
    st.image(uploaded_file, caption=f"Predicted: {predicted_class}")
    st.write(f"Prediction probabilities: {predictions}")
