import streamlit as st
from Similarity_check import load_model, resize_image, check_similarities
from PIL import Image
import tensorflow as tf





##########
import streamlit as st
import numpy as np
import cv2

# # Load the Siamese model
# model = tf.keras.models.load_model('siamese_model.h5')

# # Define a function to predict the similarity between two images
# def predict_similarity(image1, image2):
#     # Convert the images to NumPy arrays
#     image1 = np.array(image1)
#     image2 = np.array(image2)

#     # Resize the images to the same size
#     image1 = cv2.resize(image1, (224, 224))
#     image2 = cv2.resize(image2, (224, 224))

#     # Predict the similarity between the two images
#     similarity = model.predict([image1, image2])

#     return similarity

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



######
# # Load your Siamese network model
# model = load_model()

# # Define the Streamlit app
# def main():
#     st.title("Siamese Network App")
#     st.write("Upload two images and get the similarity score!")

#     # Allow users to upload images
#     image1 = st.file_uploader("Upload Image 1", type=['jpg', 'jpeg', 'png'])
#     image2 = st.file_uploader("Upload Image 2", type=['jpg', 'jpeg', 'png'])
#     print(image1)

#     if image1 and image2:
#         # Display the uploaded images
#         st.image([image1, image2], caption=["Image 1", "Image 2"], width=200)

#         # Preprocess the images
#         image_1 = resize_image(image1)
#         image_2 = resize_image(image2)

#         similarity_score = check_similarities(image_1, image_2)

#         # Display the similarity score
#         st.write("Similarity Score:", similarity_score)

# # Run the app
# if __name__ == '__main__':
#     main()
