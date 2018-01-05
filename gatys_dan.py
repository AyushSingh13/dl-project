import keras
import keras.backend as K
from keras.applications import vgg19
from keras.optimizers import Adam
from keras.layers import Input

from scipy.misc import imsave

from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np

# Image reconstruction from layer 'conv4_2'
content_layer = 'block4_conv2'
# Style reconstruction layers
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Content Image Path
content_img_path = 'img/content/golden_gate.jpg'
# Style Image Path
style_img_path = 'img/style/starry_night.jpg'
# Number of iterations
num_iterations = 1000
# Content Weight
content_weight = 1.
# Style Weight
style_weight = .00000000001
# Image size
img_size = 256
# Style image size
style_size = 256

def load_image(path, size):
    img = load_img(path, target_size=(size,size))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    x = x.reshape((img_size, img_size, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_image(x,path):
    img = deprocess_image(x)
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

# Load images
content_img = load_image(content_img_path, img_size)
style_img = load_image(style_img_path, style_size)

# Randomly initialize the generated image
generated_img = K.variable(K.random_normal(content_img.shape, stddev=0.001))
#generated_img = K.random_normal(content_img.shape, stddev=0.001)

# Load pretrained VGG19 model
# Gatys et al. specify they do not use any fully-connected layers
# Average pooling used to give better gradient flow
#model = vgg19.VGG19(include_top=False, pooling='avg')
model = vgg19.VGG19(include_top=False, pooling='avg', input_tensor=Input(tensor=generated_img))
model_layers = {layer.name: layer.output for layer in model.layers}

# Get the layers we want from VGG19
content_img_features = [model_layers[content_layer]]
style_img_features = [gram_matrix(model_layers[l]) for l in style_layers]

# Functions for getting the contents of the layers later
get_content = K.function([model.input], content_img_features)
get_style = K.function([model.input], style_img_features)

content_target = get_content([content_img])
style_target = get_style([style_img])

content_target_var = K.variable(content_target[0])
style_target_var = [K.variable(t) for t in style_target]

#model = vgg19.VGG19(include_top=False, pooling='avg', input_tensor=generated_img)
#model2_layers = {layer.name: layer.output for layer in model.layers}

# Losses
content_loss = content_loss(content_img_features[0], content_target_var)
style_loss = [style_loss(f,t) for f,t in zip(style_img_features, style_target_var)]

# Total variation loss here???

total_content_loss = K.mean(content_loss) * content_weight
total_style_loss = K.sum([K.mean(loss)*style_weight for loss in style_loss])
total_loss = K.variable(0.) + total_content_loss + total_style_loss

optimizer = Adam(lr=10)
updates = optimizer.get_updates(total_loss, [generated_img])
outputs = [total_loss, total_content_loss, total_style_loss]
step = K.function([], outputs, updates)

for i in range(num_iterations):
    print("Iteration %d of %d" % (i, num_iterations))
    res = step([])
    
    print("Content loss: %g; Style loss: %g; Total loss: %g;" % (res[1],res[2],res[0]))
    
    if i%20 == 0:
        y = K.get_value(generated_img)
        path = "output/out_%d.jpg" % i
        img = save_image(y, path)

