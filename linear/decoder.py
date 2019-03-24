# -*- coding: utf-8 -*-

import tensorflow as tf
import os

# if USE_TF_KERAS:
Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
Layer = tf.keras.layers.Layer
Model = tf.keras.models.Model
VGG19 = tf.keras.applications.vgg19.VGG19


class SpatialReflectionPadding(Layer):

    def __init__(self, **kwargs):
        super(SpatialReflectionPadding, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]+2, input_shape[2]+2, input_shape[3])

    def call(self, x):
        return tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")


def vgg_decoder(input_size=None):
    
    def _build_model(input_shape):
        x = Input(shape=input_shape, name="input")
        img_input = x
    
        # Block 3
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = tf.keras.layers.UpSampling2D()(x)
 
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = tf.keras.layers.UpSampling2D()(x)
 
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
        x = Conv2D(3, (3, 3), activation=None, padding='same', name='block1_conv2')(x)
        
        # Block 3
        model = Model(img_input, x, name='vgg19_decoder')
        return model

    input_shape=[input_size,input_size,256]
    model = _build_model(input_shape)
    
    fname = os.path.join(os.path.dirname(__file__), "vgg_decoder_31.h5")
    model.load_weights(fname, by_name=True)
    return model


if __name__ == '__main__':
    model = vgg_decoder(input_size=64)
    model.summary()
    
    import numpy as np
    torch_params = []
    for i in range(5):
        fname_b = "../bias_d{}.npy".format(i)
        fname_w = "../weight_d{}.npy".format(i)
        bias = np.load(fname_b)
        weight = np.load(fname_w)
        torch_params.append([weight, bias])
        print(weight.shape, bias.shape)

    i = 0
    for layer in model.layers:
        params = layer.get_weights()
          
        if len(params) == 2:
            weights, biases = params
            print(layer.name, weights.shape, biases.shape)
            layer.set_weights(torch_params[i])
            i+=1
    print(i)
    model.save_weights("vgg_decoder_31.h5")

    
    
    
