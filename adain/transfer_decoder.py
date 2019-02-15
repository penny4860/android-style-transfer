
# import modules
import numpy as np
import os
import glob
import tensorflow as tf
import keras

np.random.seed(1337)
from adain import MODEL_ROOT, USE_TF_KERAS
from adain.decoder import combine_and_decode_model
from adain.layers import SpatialReflectionPadding


IMG_ROOT = os.path.join("experiments", "imgs")
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


def build_mobile_combine_decoder(vgg_combine_decoder, vgg_output_block=2):
    
    vgg_output_layer_name = "b{}_output".format(vgg_output_block)
    x = vgg_combine_decoder.get_layer(vgg_output_layer_name).output

    x = SpatialReflectionPadding()(x)
    x = DepthwiseConv2D((3, 3), use_bias=False, padding='valid', name='b1_layer3_depthconv3x3')(x)
    x = BatchNormalization(name='b1_layer3_bn')(x)
    x = Activateion("relu")(x)
    x = Conv2D(64, (1, 1), use_bias=False, padding='valid', name='b1_layer2_conv1x1')(x)
    x = BatchNormalization(name='b1_layer2_bn')(x)
    x = Activateion("relu")(x)
    
    x = SpatialReflectionPadding()(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='valid', name='b1_layer1_conv3x3')(x)
    model = Model(vgg_combine_decoder.inputs, x, name='mobile_decoder')
    
    for layer in model.layers:
        if layer.name == vgg_output_layer_name:
            break
        layer.trainable = False
        print(layer.name)
    return model



if __name__ == '__main__':
    input_size = 256
    feature_size = int(input_size/8)  

    vgg_combine_decoder = combine_and_decode_model(feature_size=feature_size,
                                                   include_post_process=False)
    vgg_combine_decoder.summary()
    mobile_combine_decoder = build_mobile_combine_decoder(vgg_combine_decoder)
    mobile_combine_decoder.summary()

