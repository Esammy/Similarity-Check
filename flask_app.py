from flask import Flask, request, render_template
import numpy as np
import cv2
from Similarity_check import load_model, resize_image, check_similarities
import os


app = Flask(__name__)

def assure_path_exists(path):
  try:
      dir = os.path.dirname(path)
      if not os.path.exists(dir):
          os.makedirs(dir)
  except:
      print('Error has occured')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded images
        image1 = request.files['image1']
        image2 = request.files['image2']
        # print('the file name is:',image1.filename)
        path = 'image/jpeg/'
        assure_path_exists(path)
        
        image1.save(path)
        image2.save(path)

        # Preprocess the images
        img1 = resize_image(path+image1.filename)
        img2 = resize_image(path+image2.filename)

        # Perform inference with the Siamese model (adjust as needed)
        similarity_score = check_similarities(img1, img2)

        # Render the result template with the similarity score
        return render_template('result.html', similarity_score=similarity_score)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

