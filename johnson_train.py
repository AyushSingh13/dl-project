import os
import keras
import transform


if __name__ == '__main__':
   
    # Image Transformation Network to trainn
    transform_net = transform.model(256, "instance")
