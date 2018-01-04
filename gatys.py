from keras.preprocessing.image import load_img, img_to_array, array_to_img
import keras.backend as K
import tensorflow as tf
import numpy as np
# image_data_format -> 'channels_last'

import keras.applications.vgg19 as vgg19
from scipy.optimize import fmin_l_bfgs_b

# Image reconstruction from layer 'conv4_2'
content_layer = 'block4_conv2'

# Style reconstruciton layers
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                'block4_conv1', 'block5_conv1']

style_weight, content_weight = 1, 1

def load_image(path):
    img = np.expand_dims(img_to_array(load_img(path, target_size=(256, 256))), axis=0)
    return vgg19.preprocess_input(img)

def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(content, new):
    return 0.5 * K.sum((content - new) ** 2)

def style_loss(style, new):
    style_gram, new_gram = gram_matrix(style), gram_matrix(new)
    size = 256 ** 2
    channels = 3
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

# Match content representation
total_loss = K.variable(0.)
conv4_2_features = layers[content_layer]
content_img_features = conv4_2_features[0, :, :, :] # 1st element in input_tensor to VGG
generated_img_features = conv4_2_features[2, :, :, :] # 3rd element in input_tensor to VGG
total_loss += content_loss(content_img_features, generated_img_features)

# Build style representation
style_img_loss = K.variable(0.)
for name in style_layers:
    layer_features = layers[name]
    style_img_features = layer_features[1, :, :, :]
    generated_img_features = layer_features[2, :, :, :]
    sl = style_loss(style_img_features, generated_img_features)
    total_loss += (style_weight / len(style_layers)) * sl

loss_grads = K.gradients(total_loss, generated_img)

outputs = [total_loss]
if type(loss_grads) in {list, tuple}:
    outputs += loss_grads
else:
    outputs.append(loss_grads)

generate_func = K.function([generated_img], outputs)

class GradLoss:
    def __init__(self):
        self.loss, self.grads = None, None

    def _get_loss_and_grads(img):
        outputs = generated_func([img])
        loss = outputs[0]
        grads = np.array(outputs[1:]).flatten().astype('float64')
        return loss, grads

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self._get_loss_and_grads(x)
        self.loss = loss_value
        self.grads = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss = None
        self.grads = None
        return grad_values

grad_and_loss = GradLoss()

x = load_image('./img/content/nyc.jpg')

for i in range(5):
    print("Iteration %d:" % (i))
    print(grad_and_loss)
    x, loss_at_min, description = fmin_l_bfgs_b(grad_and_loss.loss,
                                        x, grad_and_loss.grads)
    print("Loss: %d" % (loss_at_min))
    img_out = deprocess_image(x)
    img_name = 'nyc_starrynight_iteration%d.jpg' % i
    imsave(img_name, img_out)
    