import posixpath
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # turn off tf-cpu logging message
import numpy as np
import cv2
from tensorflow import keras


MAPPING = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B', 
                12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 
                23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 
                34:'Y', 35:'Z', 36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 41:'g', 42:'h', 43:'n', 44:'q', 
                45:'r', 46:'t'}


parser = argparse.ArgumentParser(description='Read images from a directory.')
parser.add_argument('--input', type=str, help='Path to the directory with images')
args = parser.parse_args()


def preprocess_image(image):
    """ 
    Converts image to for model
    input image of shape (n, n, n) to (28, 28, 1) 
    """
    image = cv2.resize(image, (28, 28))
    # in case image does not have 3rd dimension cv2.cvtColor throws error
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = keras.utils.normalize(image, axis = 1)
    
    return image


def get_prediction(model, image):
    
    p = model.predict(image.reshape(1, 28, 28, 1), verbose=0)
    # get index of class with maximum probability
    idx_max = p[0].argmax()

    return idx_max

def read_images(image_dir):
    images, paths = [], []
    """Read jpeg/jpg/png images from image_dir"""
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            # convert to posix path
            image_path = posixpath.normpath(posixpath.join(image_dir, filename))
            image = cv2.imread(image_path)
            image = np.array(image)
            image = preprocess_image(image)
            images.append(image)
            paths.append(image_path)
    
    return images, paths


if __name__ == '__main__':

    image_dir = args.input

    images, paths = read_images(image_dir)

    model = keras.models.load_model('./model/VIN_model.h5')
    
    predictions = []
    for img in images:
        prediction = get_prediction(model, img)
        predictions.append(ord(MAPPING[prediction]))

    for code, path in zip(predictions, paths):
        print(f'{code:03d}, {path}')