# -*- coding: utf-8 -*-

import tensorflow as tf
import os

# if USE_TF_KERAS:
Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
BatchNormalization = tf.keras.layers.BatchNormalization
Activateion = tf.keras.layers.Activation
MaxPooling2D = tf.keras.layers.MaxPooling2D
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


def vgg_encoder(input_size=256):
    
    def _build_model(input_shape):
        x = Input(shape=input_shape, name="input")
        img_input = x
    
        # Block 1
        # x = VggPreprocess()(x)
        x = Conv2D(3, (1, 1), activation=None, padding='same', name='block1_conv0')(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
           
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
           
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)


#         x = SpatialReflectionPadding()(x)
#         x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv2')(x)
#         x = SpatialReflectionPadding()(x)
#         x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv3')(x)
#         x = SpatialReflectionPadding()(x)
#         x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv4')(x)
#         x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#         
#         # Block 4
#         x = SpatialReflectionPadding()(x)
#         x = Conv2D(512, (3, 3), activation='relu', padding='valid', name="output")(x)
        model = Model(img_input, x, name='vgg19')
        return model

    input_shape=[input_size,input_size,3]
    model = _build_model(input_shape)
    fname = os.path.join(os.path.dirname(__file__), "vgg_31.h5")
    model.load_weights(fname, by_name=True)
    return model


if __name__ == '__main__':
    import numpy as np
    model = vgg_encoder(input_size=None)
#     model.summary()
#     
    torch_params = []
    for i in range(6):
        fname_b = "../bias_{}.npy".format(i)
        fname_w = "../weight_{}.npy".format(i)
        bias = np.load(fname_b)
        weight = np.load(fname_w)
        torch_params.append([weight, bias])
        print(weight.shape, bias.shape)
#     
#     
    i = 0
    for layer in model.layers:
        params = layer.get_weights()
          
        if len(params) == 2:
            weights, biases = params
            print(layer.name, weights.shape, biases.shape)
            layer.set_weights(torch_params[i])
            i+=1
     
    model.save_weights("vgg_31.h5")
#     model.load_weights("vgg_31.h5")
#     y = model.predict(np.zeros((1,512,512,3)))
#     print(y.shape)
    
    
