import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    tf.keras.backend.clear_session()  # Clear the TensorFlow graph
    model = tf.keras.models.load_model('newer_newer_model_improved_improved.h5')
    return model

model = load_model()

st.title("Malaria Cell Image Classification by Jerico Delos Reyes")
st.write("CPE 019-CPE32S6 - Emerging Technologies 2 in CpE")
st.write("Final Exam: Model Deployment in the Cloud")
st.write(
        """

        Select images to upload on this link: https://drive.google.com/drive/folders/16dEDVexhQd8pHusJLGhChmVW9wCqWdq7?usp=sharing
        """
    )

st.write(
        """
        The goal of Malaria Cell Image Classification is to create a CNN model that accurately categorizes microscopic blood cell pictures as infected or uninfected, automating the diagnosis of malaria. This improves efficiency, makes early diagnosis possible, and lessens the need for manual examination, which eventually results in timely treatment and improved healthcare outcomes.
        """
    )


file = st.file_uploader("Choose photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (128, 128)

    # Resize the image to the expected input shape of the model
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)

    # Convert the image to grayscale if necessary
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to 3-channel image

    # Reshape the image to add a channel dimension
    img_reshape = img.reshape((1,) + img.shape)

    # Make predictions using the Keras model
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Parasitized', 'Uninfected']
    string1 = "The cell is " + class_names[np.argmax(prediction)] + "."
    string2= "Kindly take note that this model is not entirely precise."
    st.success(string1)
    st.success(string2)
st.write(
        """
        Github Repository: https://github.com/jericojr23/streamlit-Delos_Reyes.git

        
        The training is done on this IPYNB file: https://drive.google.com/file/d/1VEwA5ICuzPYLfi-LfEh6TARDPJ6zIW0G/view?usp=sharing
        """
    )