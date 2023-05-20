import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
    tf.keras.backend.clear_session()  # Clear the TensorFlow graph
    model = tf.keras.models.load_model('newer_newer_model_final.h5')
    return model

model = load_model()

st.write("""
# Malaria Cell Image Classification by Jerico Delos Reyes"""
)
file=st.file_uploader("Choose plant photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(128,128)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Parasitized', 'Uninfected']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)