import streamlit as st
from Similarity_check import load_model, resize_image, check_similarities
from PIL import Image
import tensorflow as tf





##########
import streamlit as st
import numpy as np
import cv2


# Create a Streamlit app
st.title('Siamese Image Similarity App')

# Upload two images
image1 = st.file_uploader('Upload Image 1')
image2 = st.file_uploader('Upload Image 2')

# If both images have been uploaded
if image1 and image2:
    # Convert the images to NumPy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)

    # Resize the images to the same size
    image1 = resize_image(image1)
    image2 = resize_image(image2)

    # Predict the similarity between the two images
    similarity = check_similarities(image1, image2)

    # Display the similarity score
    st.write('Similarity score:', similarity)

    # Display the two images
    st.image(image1, caption='Image 1')
    st.image(image2, caption='Image 2')

