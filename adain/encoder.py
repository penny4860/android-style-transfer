# -*- coding: utf-8 -*-

import tensorflow as tf
import keras
import os

from adain import USE_TF_KERAS
from adain.layers import VggPreprocess, SpatialReflectionPadding
from adain import MODEL_ROOT

VGG_ENCODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_encoder.h5")
MOBILE_ENCODER_H5 = os.path.join(MODEL_ROOT, "h5", "mobile_encoder.h5")


if USE_TF_KERAS:
    Input = tf.keras.layers.Input
    Conv2D = tf.keras.layers.Conv2D
    DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
    BatchNormalization = tf.keras.layers.BatchNormalization
    Activateion = tf.keras.layers.Activation
    MaxPooling2D = tf.keras.layers.MaxPooling2D
    Layer = tf.keras.layers.Layer
    Model = tf.keras.models.Model
    VGG19 = tf.keras.applications.vgg19.VGG19
else:
    Input = keras.layers.Input
    Conv2D = keras.layers.Conv2D
    DepthwiseConv2D = keras.layers.DepthwiseConv2D
    BatchNormalization = keras.layers.BatchNormalization
    Activateion = keras.layers.Activation
    MaxPooling2D = keras.layers.MaxPooling2D
    Layer = keras.layers.Layer
    Model = keras.models.Model
    VGG19 = keras.applications.vgg19.VGG19


def extract_feature_model(input_size=256, output_layer="block2_conv2"):
    def _build_model(input_shape):
        x = Input(shape=input_shape, name="input")
        img_input = x
        
        x = VggPreprocess()(x)
        base_model = VGG19(input_shape=input_shape, include_top=False)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(output_layer).output)
        model.load_weights("models/h5/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)
        x = model(x)

        feat_model = Model(img_input, x, name='vgg19_extractor')
        return feat_model

    input_shape=[input_size,input_size,3]
    model = _build_model(input_shape)
    return model


def vgg_encoder(input_size=256,
                h5_fname=VGG_ENCODER_H5):
    
    def _build_model(input_shape):
        x = Input(shape=input_shape, name="input")
        img_input = x
    
        # Block 1
        x = VggPreprocess()(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # Block 2
        x = SpatialReflectionPadding()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        # Block 3
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv2')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv3')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name="output")(x)
        model = Model(img_input, x, name='vgg19')
        return model

    input_shape=[input_size,input_size,3]
    model = _build_model(input_shape)
    model.load_weights(h5_fname)
    return model


def mobile_encoder(input_size=256, h5_fname=MOBILE_ENCODER_H5):

    input_shape=[input_size,input_size,3]
    
    x = Input(shape=input_shape, name="input")
    img_input = x

    # Block 1
    x = VggPreprocess()(x)
    x = Conv2D(32, (3, 3), strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    # (112,112,32)

    x = DepthwiseConv2D((3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(64, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    # (112,112,64)

    x = DepthwiseConv2D((3, 3), strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    # (56,56,64)
    x = Conv2D(128, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    # (56,56,128)

    x = DepthwiseConv2D((3, 3), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(128, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    # (56,56,128)

    x = DepthwiseConv2D((3, 3), strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    # (28,28,128)
    x = Conv2D(256, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    # (28,28,256)

    x = DepthwiseConv2D((3, 3), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(256, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    # (28,28,256)

    x = DepthwiseConv2D((3, 3), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(512, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    # (28,28,512)

    x = DepthwiseConv2D((3, 3), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(512, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu", name="output")(x)

    model = Model(img_input, x, name='vgg19_light')
    model.load_weights(h5_fname)
    return model


if __name__ == '__main__':
    model = vgg_encoder()
    light_model = mobile_encoder()
    mobilenet = tf.keras.applications.mobilenet.MobileNet(input_shape=(224,224,3))
    print("======================================================")
    conv_params = []
    bn_params = []
    for layer in mobilenet.layers:
        params = layer.get_weights()
        if len(params) == 1:
            conv_params.append(params)
        if len(params) == 4:
            bn_params.append(params)
    print("======================================================")

    ci = 0
    bi = 0    
    for layer in light_model.layers:
        params = layer.get_weights()
        if len(params) == 1:
            print(layer.name, params[0].shape, conv_params[ci][0].shape)
            layer.set_weights(conv_params[ci])
            ci += 1
        if len(params) == 4:
            print(layer.name, params[0].shape, bn_params[bi][0].shape)
            layer.set_weights(bn_params[bi])
            bi += 1
    print(ci, bi)
    light_model.save_weights("mobile_init.h5")
    

