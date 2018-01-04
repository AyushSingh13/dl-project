from keras.preprocessing.image import load_img, img_to_array, array_to_img
import keras.backend as K
import tensorflow as tf
import numpy as np
# image_data_format -> 'channels_last'

import keras.applications.vgg19 as vgg19

style_weight, content_weight = 1, 1

def load_image(path):
    return np.expand_dims(img_to_array(load_img(path, target_size=(256, 256))), axis=0)

def content_loss(content, new):
    return 0.5 * K.sum((content - new) ** 2)

def style_loss(style, new):
    style_gram, new_gram = gram_matrix(style), gram_matrix(new)
    size = 256 ** 2
    return K.sum(K.square(style_gram - new_gram)) / (4. * (channels ** 2) * (size ** 2))

def total_loss(content, style, new):
    content = content_weight * content_loss(content, new)
    style = style_weight * style_loss(style, new)
    return content + style

def gram_matrix(img):
    f = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
    return K.dot(f, K.transpose(f))

content_img = K.variable(load_image('./img/content/nyc.jpg'))
style_img = K.variable(load_image('./img/style/starry_night.jpg'))
generated_img = K.placeholder((1, 256, 256, 3))

assert content_img.shape == style_img.shape == generated_img.shape == (1, 256, 256, 3)

input_tensor = K.concatenate([content_img, style_img, generated_img], axis=0)

# Gatys et al. state they do not use any fully-connected layers
# Average pooling used to give better gradient flow
model = vgg19.VGG19(input_tensor=input_tensor, include_top=False, pooling='avg')

layers = {layer.name: layer.output for layer in model.layers}

# The layers used for content reconstructions
feature_maps = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Train