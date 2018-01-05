from keras.applications import vgg19
import keras.backend as K
from scipy.misc import imsave
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np

def load_image(path, size, interp):
    img = load_img(path, target_size=(size,size), interpolation=interp)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x, size):
    x = x.reshape((size, size, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_image(x,path,size):
    img = deprocess_image(x, size)
    imsave(path, img)

def gram_matrix(x):
    # Make 3d from 4d
    f = K.reshape(x, K.shape(x)[1:4])
    f = K.batch_flatten(K.permute_dimensions(f, (2,1,0)))
    gram = K.dot(f, K.transpose(f))
    return gram

def content_loss(x, t):
    return K.mean(K.square(t - x), axis=(1,2,3))

def style_loss(x, t):
    return K.mean(K.square(t - x), axis=(0,1))

