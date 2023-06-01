from tensorflow import keras
import cv2

# Load the model
def load_model():
    print('Loading model...')
    return keras.models.load_model("Similarity_test.h5")


def check_similarities(img_A, img_B):
    try:
        model=load_model()
        return model.predict([img_A.reshape((1, 128, 128, 3)), 
            img_B.reshape((1, 128, 128, 3))]).flatten()[0]>0.5
    except:
        return "Image returned None"

def resize_image(image, face=False):
    if face==True:
        print('Extracting and resizing face...')
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        imag = cv2.imread(image)
        gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print ("Found {0} faces... Note: only one face is supported for now.".format(len(faces)))
        if len(faces)>1:
            return None
    
        try:
            if len(faces) == 1:
                for (x, y, w, h) in faces:
                    cv2.rectangle(imag, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    image_frame = imag[y:y + h, x:x + w]

                    new_size = (128, 128)
                    return cv2.resize(image_frame, new_size)
            elif len(faces) > 1:
                print(f'Found {len(faces)} faces')
                return None
        except:
            print('No face found!')
            return None
    else:
        print('Resizing image...')
        image = cv2.imread(image)
        new_size = (128, 128)
        return cv2.resize(image, new_size)
    
if __name__ == "__main__":
    image_1 = resize_image('Sammy1.jpg', face=True)
    image_2 = resize_image('Nothing.jpg', face=True)

    check = check_similarities(image_1, image_2)
    if check == True:
        print('Match found!')
    elif check == False:
        print('Match not found!')
    else:
        print('Something is wrong check your input image')
