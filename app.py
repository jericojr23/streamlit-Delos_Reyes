import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
    tf.keras.backend.clear_session()  # Clear the TensorFlow graph
    model = tf.keras.models.load_model('newer_newer_model.h5')
    return model

model = load_model()

st.write("""
# Malaria Cell Image Classification by Jerico Delos Reyes"""
)
file=st.file_uploader("Choose plant photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
# def import_and_predict(image_data,model):
#     size=(128,128)
#     image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
#     img=np.asarray(image)
#     img_reshape=img[np.newaxis,...]
#     prediction=model.predict(img_reshape)
#     return prediction

import cv2
from PIL import Image, ImageOps
import numpy as np

import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

def import_and_predict(image_data, model):
    size = (128, 128)
    
    # Resize the image to the expected input shape of the model
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
    
    # Convert the image to grayscale if necessary
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Ensure the image has 3 channels
    img_reshape = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    # Reshape the image to match the input shape of the model
    img_reshape = img_reshape.reshape((1,) + img_reshape.shape)
    
    # Clear the previous TensorFlow graph
    tf.keras.backend.clear_session()
    
    # Create a new TensorFlow graph
    graph = tf.Graph()
    
    with graph.as_default():
        # Load the model within the new graph
        with tf.Session(graph=graph):
            # Make predictions using the Keras model
            prediction = model.predict(img_reshape)
    
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