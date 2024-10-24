from PIL import Image
import numpy as np


def load_image(filename):
    # convert to 8bit greyscale
    img = Image.open(filename).convert('L') 
    img = img.resize((28,28))
    # convert to numpy array
    img_array = np.array(img)
    # flatens into single column vector (781,1)
    img_data = img_array.reshape(784,1)
    # normalize the pixels 0->255 -->> 0 ->1 greyscale
    img_data = img_data / 255.0
    return img_data.flatten()