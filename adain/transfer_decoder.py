
# import modules
import numpy as np
import os
import tensorflow as tf
import keras

np.random.seed(1337)
from adain import MODEL_ROOT, USE_TF_KERAS
from adain.decoder import combine_and_decode_model
from adain.layers import SpatialReflectionPadding
from adain.layers import PostPreprocess


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
    Add = tf.keras.layers.Add
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
    Add = keras.layers.Add


def depthwise_separable_block(x, out_filters, block_idx, layer_idx):
    x = SpatialReflectionPadding()(x)
    x = DepthwiseConv2D((3, 3), use_bias=False, padding='valid',
                        name='b{}_layer{}_depthconv3x3'.format(block_idx, layer_idx))(x)
    x = BatchNormalization(fused=False, name='b{}_layer{}_bn_d'.format(block_idx, layer_idx))(x)
    x = Activateion("relu")(x)
    x = Conv2D(out_filters, (1, 1), use_bias=False, padding='valid', name='b{}_layer{}_conv1x1'.format(block_idx, layer_idx))(x)
    x = BatchNormalization(fused=False, name='b{}_layer{}_bn_p'.format(block_idx, layer_idx))(x)
    x = Activateion("relu")(x)
    return x


def build_mobile_b4(x):
    x = depthwise_separable_block(x, 256, block_idx=4, layer_idx=1)
    x = UpSampling2D(name="b4_output")(x)
    return x

def build_mobile_b3(x):
    shortcut = x
    x = depthwise_separable_block(x, 256, block_idx=3, layer_idx=4)
    x = Add()([shortcut, x])

    shortcut = x
    x = depthwise_separable_block(x, 256, block_idx=3, layer_idx=3)
    x = Add()([shortcut, x])

    shortcut = x
    x = depthwise_separable_block(x, 256, block_idx=3, layer_idx=2)
    x = Add()([shortcut, x])

    x = depthwise_separable_block(x, 128, block_idx=3, layer_idx=1)
    x = UpSampling2D(name="b3_output")(x)
    return x

def build_mobile_b2(x):
    x = depthwise_separable_block(x, 128, block_idx=2, layer_idx=2)
    x = depthwise_separable_block(x, 64, block_idx=2, layer_idx=1)
    x = UpSampling2D(name="b2_output")(x)
    return x

def build_mobile_b1(x):
    x = depthwise_separable_block(x, 64, block_idx=1, layer_idx=2)
    x = SpatialReflectionPadding(name="b1_layer1_pad")(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='valid', name='b1_layer1_conv3x3')(x)
    return x


def build_mobile_combine_decoder(feature_size=32, num_new_blocks=4, include_post_process=False):
    
    vgg_combine_decoder = combine_and_decode_model(feature_size=feature_size)
    
    if num_new_blocks == 1:
        vgg_output_layer_name = "b2_output"
        x = vgg_combine_decoder.get_layer(vgg_output_layer_name).output
        x = build_mobile_b1(x)
        if include_post_process:
            x = PostPreprocess(name="output")(x)
        model = Model(vgg_combine_decoder.inputs, x, name='mobile_decoder')
    elif num_new_blocks == 2:
        vgg_output_layer_name = "b3_output"
        x = vgg_combine_decoder.get_layer(vgg_output_layer_name).output
        x = build_mobile_b2(x)
        x = build_mobile_b1(x)
        if include_post_process:
            x = PostPreprocess(name="output")(x)
        model = Model(vgg_combine_decoder.inputs, x, name='mobile_decoder')
    elif num_new_blocks == 3:
        vgg_output_layer_name = "b4_output"
        x = vgg_combine_decoder.get_layer(vgg_output_layer_name).output
        x = build_mobile_b3(x)
        x = build_mobile_b2(x)
        x = build_mobile_b1(x)
        if include_post_process:
            x = PostPreprocess(name="output")(x)
        model = Model(vgg_combine_decoder.inputs, x, name='mobile_decoder')
    elif num_new_blocks == 4:
        vgg_output_layer_name = "adain"
        x = vgg_combine_decoder.get_layer(vgg_output_layer_name).output
        x = build_mobile_b4(x)
        x = build_mobile_b3(x)
        x = build_mobile_b2(x)
        x = build_mobile_b1(x)
        if include_post_process:
            x = PostPreprocess(name="output")(x)
        model = Model(vgg_combine_decoder.inputs, x, name='mobile_decoder')

    print("=================================================================")
    for layer in model.layers:
        if layer.name == vgg_output_layer_name:
            break
        layer.trainable = False
        print(layer.name)
    print("==============FROZEN LAYERS =====================================")
    return model


if __name__ == '__main__':
    input_size = 256
    feature_size = int(input_size/8)  

    mobile_combine_decoder = build_mobile_combine_decoder(feature_size, include_post_process=True)
    mobile_combine_decoder.summary()

# 4.
# Total params: 420,611
# Trainable params: 414,083
# Non-trainable params: 6,528

# 3. 
# Total params: 1,461,763
# Trainable params: 276,867
# Non-trainable params: 1,184,896

# 2.
# Total params: 3,280,771
# Trainable params: 34,435
# Non-trainable params: 3,246,336

# 1.
# Total params: 3,473,475
# Trainable params: 6,659
# Non-trainable params: 3,466,816