# -*- coding: utf-8 -*-

import tensorflow as tf
import keras
import os

from adain import USE_TF_KERAS, MODEL_ROOT
from adain.layers import PostPreprocess, SpatialReflectionPadding, AdaIN

VGG_DECODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_decoder.h5")

if USE_TF_KERAS:
    Input = tf.keras.layers.Input
    Conv2D = tf.keras.layers.Conv2D
    Model = tf.keras.models.Model
    UpSampling2D = tf.keras.layers.UpSampling2D
    Layer = tf.keras.layers.Layer
    SeparableConv2D = tf.keras.layers.SeparableConv2D
    DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
    BatchNormalization = tf.keras.layers.BatchNormalization
    Activateion = tf.keras.layers.Activation

else:
    Input = keras.layers.Input
    Conv2D = keras.layers.Conv2D
    Model = keras.models.Model
    UpSampling2D = keras.layers.UpSampling2D
    Layer = keras.layers.Layer
    SeparableConv2D = keras.layers.SeparableConv2D
    DepthwiseConv2D = keras.layers.DepthwiseConv2D
    BatchNormalization = keras.layers.BatchNormalization
    Activateion = keras.layers.Activation


def build_vgg_decoder(input_features, include_post_process):
    
    # (32,32,512)
    x = input_features
    
    # Block 4
    x = SpatialReflectionPadding()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='vgg_decoder_b4_layer1_conv3x3')(x)
    x = UpSampling2D(name="b4_output")(x)

    # Block 3
    x = SpatialReflectionPadding()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='vgg_decoder_b3_layer4_conv3x3')(x)
    x = SpatialReflectionPadding()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='vgg_decoder_b3_layer3_conv3x3')(x)
    x = SpatialReflectionPadding()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='vgg_decoder_b3_layer2_conv3x3')(x)
    x = SpatialReflectionPadding()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='vgg_decoder_b3_layer1_conv3x3')(x)
    x = UpSampling2D(name="b3_output")(x)

    # Block 2
    x = SpatialReflectionPadding()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='vgg_decoder_b2_layer2_conv3x3')(x)
    x = SpatialReflectionPadding()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='vgg_decoder_b2_layer1_conv3x3')(x)
    x = UpSampling2D(name="b2_output")(x)

    # Block 1
    x = SpatialReflectionPadding()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='vgg_decoder_b1_layer2_conv3x3')(x)
    x = SpatialReflectionPadding()(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='valid', name='vgg_decoder_b1_layer1_conv3x3')(x)
    if include_post_process:
        x = PostPreprocess(name="output")(x)
    return x


def combine_and_decode_model(feature_size=32,
                             alpha=1.0,
                             include_post_process=True,
                             h5_fname=VGG_DECODER_H5):
    
    input_shape=[feature_size,feature_size,512]
    c_feat_input = Input(shape=input_shape, name="input_c")
    s_feat_input = Input(shape=input_shape, name="input_s")
    
    x = AdaIN(alpha, name="adain")([c_feat_input, s_feat_input])
    x = build_vgg_decoder(x, include_post_process)

    model = Model([c_feat_input, s_feat_input], x, name='decoder')
    model.load_weights(h5_fname)
    return model


def vgg_decoder(input_shape=[None,None,512]):
    input_layer = Input(shape=input_shape, name="input")
    x = build_vgg_decoder(input_layer)
    model = Model(input_layer, x, name='vgg_decoder')
    return model


if __name__ == '__main__':
    # Total params: 3,505,219
    # Total params:   408,003
    model = combine_and_decode_model(feature_size=32)
    model.summary()

