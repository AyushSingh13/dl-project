from keras.applications import vgg19
import keras.backend as K
from scipy.misc import imsave
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from PIL import Image
import numpy as np

def load_image(path, size, interp):
    img = load_img(path)#, target_size=(size,size)), interpolation=interp)
    if interp == "nearest":
        resample = Image.NEAREST
    if interp == "bicubic":
        resample = Image.BICUBIC
    img.thumbnail((size,size), resample)
    size = (img.height, img.width, 3)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img, size

def deprocess_image(x, size):
    x = x.reshape(size)
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_image(x,path,size):
    x = x.copy()
    img = deprocess_image(x, size)
    imsave(path, img)

def gram_matrix(x, norm):
    # Make 3d from 4d
    shape = K.shape(x)
    H, W, C = shape[1], shape[2], shape[3]
    f = K.reshape(x, (H,W,C))
    f = K.batch_flatten(K.permute_dimensions(f, (2,0,1)))
    gram = K.dot(f, K.transpose(f))
    if norm:
        gram = gram / K.cast(C*H*W, x.dtype)
    return gram

def content_loss(x, t):
    #return K.mean(K.square(t - x), axis=(1,2,3))
    x = K.reshape(x, K.shape(x)[1:4])
    t = K.reshape(t, K.shape(t)[1:4])
    return K.sum(K.square(t - x))

def style_loss(x, t, size):
    return K.sum(K.square(t - x)) / (4. * (size[2] ** 2) * ((size[0]*size[1]) ** 2))

def total_variation_loss(x):
    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = K.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return K.sum(a + b, axis=(1, 2, 3))
