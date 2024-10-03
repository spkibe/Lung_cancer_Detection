import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# import cv2

# Function to load the trained model
@st.cache_resource
def load_trained_model():
    conv_base = Xception(weights=None, include_top=False, input_shape=(299, 299, 3))

    model = Sequential([
        conv_base,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.load_weights('C_CT_S.keras')  # Make sure to update the path to your trained model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess the uploaded image
def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to compute Grad-CAM
# Grad-CAM function
# def get_gradcam_heatmap(model, img_array, last_conv_layer_name="block14_sepconv2_act"):
#     # Call the model to ensure the layers are initialized
#     _ = model.predict(img_array)  # Make a prediction to initialize model's state
    
#     # Create a model that maps the input image to the activations of the last conv layer and the predictions
#     grad_model = tf.keras.models.Model(
#         [model.input], 
#         [model.get_layer("xception").get_layer(last_conv_layer_name).output, model.output]
#     )

#     # Compute the gradient of the top predicted class for the input image
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         predicted_class = tf.argmax(predictions[0])
#         loss = predictions[:, predicted_class]

#     # Compute the gradients of the predicted class with respect to the feature map
#     grads = tape.gradient(loss, conv_outputs)

#     # Compute the mean intensity of the gradient over each feature map channel
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     # Multiply each channel by the average gradient
#     conv_outputs = conv_outputs[0]
#     conv_outputs *= pooled_grads

#     # Generate the heatmap by averaging over all channels
#     heatmap = tf.reduce_mean(conv_outputs, axis=-1)

#     # Normalize the heatmap
#     heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()


# # Function to overlay heatmap on image
# def overlay_heatmap(heatmap, img, alpha=0.4, colormap=cv2.COLORMAP_JET):
#     # Resize heatmap to match the size of the image
#     heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))

#     # Convert the heatmap to RGB using colormap
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, colormap)

#     # Convert PIL image to NumPy array and merge heatmap
#     img = np.array(img)
#     superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

#     # Return as PIL image
#     return Image.fromarray(superimposed_img)


# # Function to display Grad-CAM heatmap
# def display_gradcam(img_path, heatmap, alpha=0.4, cmap=cv2.COLORMAP_JET):
#     img = image.load_img(img_path)
#     img_array = image.img_to_array(img)
    
#     # Resize heatmap to match the image size
#     heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    
#     # Convert the heatmap to RGB
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cmap)
    
#     # Superimpose the heatmap on the original image
#     superimposed_img = heatmap * alpha + img_array
#     superimposed_img = np.uint8(superimposed_img)

#     # Display the superimposed image
#     plt.imshow(superimposed_img)
#     plt.axis('off')
#     plt.show()

# Load the trained model
model = load_trained_model()

# Streamlit app interface
st.title("Lung Cancer Classification Web App")

# 1. Model Performance Section (displayed before prediction)
st.header("Model Performance")

# Training/validation loss and accuracy plots
st.subheader("Training vs. Validation Loss")
st.image("training-loss.png", caption="Training and Validation Loss")

# st.subheader("Training vs. Validation Accuracy")
# st.image("training_acc.png", caption="Training and Validation Accuracy")
# confusion matrix
st.subheader("Confusion Matrix and classification Report")
st.image("confusion-matrix.png", caption="Confusion matrix")

st.image("classification-report.png", caption="Classification  Report")

# Brief description of model performance
st.write("This model was trained using a deep learning architecture based on the Xception model. The plots above show the model's performance in terms of training and validation loss, as well as training and validation accuracy.")

st.write("""
- The **Training Loss** measures how well the model fits the training data over each epoch.
- The **Validation Loss** evaluates how well the model generalizes to unseen data (validation set).
- The **Accuracy** curves show the model's performance in making correct predictions during training and validation.
""")

# Display model performance metrics (for demonstration)
st.write("- **Final Training Accuracy**: 100%")
st.write("- **Final Validation Accuracy**: 98.25%")
st.write("- **Final Training Loss**: 0.0669")
st.write("- **Final Validation Loss**: 0.0936")

# 2. Image Upload and Prediction
st.header("Lung Cancer Classification")

st.write("Upload a lung CT scan image to predict its classification:")

# File uploader for lung CT scan image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Patient Information (optional)
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", min_value=0, max_value=120)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female", "Other"])
smoking_history = st.sidebar.selectbox("Smoking History", options=["Never Smoked", "Former Smoker", "Current Smoker"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and make prediction
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)[0][0]

    # Display classification result with probability score
    st.subheader("Prediction Outcome")
    if prediction < 0.5:
        st.write(f"Prediction: **Malignant** (Confidence: {prediction * 100:.2f}%)")
    else:
        st.write(f"Prediction: **Benign** (Confidence: {(1 - prediction) * 100:.2f}%)")

    if prediction < 0.5:
        st.write("It is advisable to pay a visit to a specialist clinic")

    # # Generate Grad-CAM heatmap
    # st.subheader("Explainability Insights")
    # st.write("Generating Grad-CAM heatmap...")
    # # Preprocess image (img_array should be your preprocessed image array)
    # # Preprocess the image
    # img_array = preprocess_image(img)

    # # Generate Grad-CAM heatmap
    # heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name="block14_sepconv2_act")

    # # Display Grad-CAM heatmap on the original image
    # display_gradcam(img, heatmap)

    # # Risk Stratification
    # st.header("Risk Stratification")
    # if prediction < 0.2:
    #     st.write("Risk Level: **Low**")
    #     st.write("Recommendation: No immediate action needed.")
    # elif prediction < 0.7:
    #     st.write("Risk Level: **Moderate**")
    #     st.write("Recommendation: Schedule follow-up imaging and consult a specialist.")
    # else:
    #     st.write("Risk Level: **High**")
    #     st.write("Recommendation: Immediate medical consultation and further tests recommended.")

    # # Model Interpretation and Reliability
    # st.header("Model Interpretation & Reliability")
    # st.write("**Uncertainty Measure**: Low uncertainty in this prediction.")
    # st.write("**Disclaimer**: This model is a support tool and not a substitute for professional medical advice.")
    
    # # Interactive user controls for uploading new data
    # st.header("Interactive User Controls")
    # st.write("You can upload new images for analysis or filter past predictions based on various parameters.")
