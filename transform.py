from keras.models import Model
from keras.layers import Input, Activation, Lambda, Conv2D, Conv2DTranspose, BatchNormalization
from keras_contrib.layers import InstanceNormalization
from keras.layers.merge import add

# Model for Image Transformation Network
# Roughly following architecture by Johnson et al
# Normalization can be modified to be none, instance or batch
# Reflection Padding is *not* implemented

def model(size=256, normalization="instance"):

    input_shape = (size, size, 3)

    image = Input(shape=input_shape)
    
    conv1 = conv(image, 32,  (9,9), (1,1), normalization=normalization)
    conv2 = conv(conv1, 64,  (3,3), (2,2), normalization=normalization)
    conv3 = conv(conv2, 128,  (3,3), (2,2), normalization=normalization)

    resid1 = residual_block(conv3, normalization=normalization)
    resid2 = residual_block(resid1, normalization=normalization)
    resid3 = residual_block(resid2, normalization=normalization)
    resid4 = residual_block(resid3, normalization=normalization)
    resid5 = residual_block(resid4, normalization=normalization)

    convt1 = convt(resid5, 64, (3,3), (2,2), normalization=normalization)
    convt2 = convt(convt1, 32, (3,3), (2,2), normalization=normalization)
    convt3 = convt(convt2, 3, (9,9), (2,2), normalization=normalization, relu=False)
    
    y = Activation("tanh")(convt3)
    y = Lambda(lambda x: x*150)(y)

    return Model(input=image, output=y)

def conv(x, filters, kernel_size=(3,3), strides=(1,1), relu=True, normalization="instance"):
    y = Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    if normalization == "batch":
        y = BatchNormalization(axis=-1)(y)
    if normalization == "instance":
        y = InstanceNormalization(axis=-1)(y)
    if relu:
        y = Activation("relu")(y)
    return y

def convt(x, filters, kernel_size=(3,3), strides=(2,2), relu=True, normalization="instance"):
    y = Conv2DTranspose(filters, kernel_size, strides=strides, padding="same")(x)
    if normalization == "instance":
        y = InstanceNormalization(axis=-1)(y)
    if relu:
        y = Activation("relu")(y)
    return y

def residual_block(x, kernel_size=(3,3), normalization="instance"):
    a = conv(x, 128, kernel_size, (1,1), normalization=normalization)
    b = conv(a, 128, kernel_size, (1,1), normalization=normalization, relu=False)
    return add([a, b])


