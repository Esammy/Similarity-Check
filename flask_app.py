from flask import Flask, request, render_template
import numpy as np
import cv2
from Similarity_check import resize_image, check_similarities
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
        
        image1.save(path+image1.filename)
        image2.save(path+image2.filename)

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


# from flask import Flask, render_template, request
# import os

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Get the uploaded image
#         image = request.files['image']
#         filename = image.filename

#         # Save the image to a temporary directory
#         temp_dir = 'temp/'
        
#         os.makedirs(temp_dir, exist_ok=True)
#         image_path = os.path.join(temp_dir, filename)
#         print(temp_dir + filename)
#         image.save(image_path)
#         image_name = temp_dir+filename

#         # Render the preview template with the image path
#         return render_template('index.html', image_name=image_name)

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
