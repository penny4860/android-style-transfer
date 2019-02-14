
# import modules
import numpy as np
import os
import glob
import tensorflow as tf
import keras
import argparse

np.random.seed(1337)
from adain import MODEL_ROOT, USE_TF_KERAS
from adain.encoder import vgg_encoder
from adain.decoder import combine_and_decode_model
from adain.generator import CombineBatchGenerator, create_callbacks


DEFAULT_IMG_ROOT = os.path.join("experiments", "imgs")
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_INIT_WEIGHTS = None
DEFAULT_VGG_ENCODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_encoder.h5")
DEFAULT_VGG_DECODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_decoder.h5")


argparser = argparse.ArgumentParser(description='Train mobile decoder')
argparser.add_argument('-i',
                       '--image_root',
                       default=DEFAULT_IMG_ROOT,
                       help='images directory')
argparser.add_argument('-w',
                       '--weights_init',
                       default=DEFAULT_INIT_WEIGHTS,
                       help='learning rate')
argparser.add_argument('-b',
                       '--batch_size',
                       default=DEFAULT_BATCH_SIZE,
                       type=int,
                       help='batch size')
argparser.add_argument('-l',
                       '--learning_rate',
                       default=DEFAULT_LEARNING_RATE,
                       type=float,
                       help='learning rate')


from adain.layers import SpatialReflectionPadding
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


def build_model(vgg_combine_decoder):
    x = vgg_combine_decoder.get_layer("spatial_reflection_padding_17").output

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
        if layer.name == "spatial_reflection_padding_17":
            break
        layer.trainable = False
        print(layer.name)
    return model


def loss_func(y_true, y_pred):
    # 1. activate prediction & truth tensor
    style_loss = tf.losses.mean_squared_error(y_true, y_pred)
    tv_loss = tf.reduce_mean(tf.image.total_variation(y_pred))
    loss = style_loss + 1e-8*tv_loss
    return loss


if __name__ == '__main__':
    args = argparser.parse_args()
    
    input_size = 256
    decoder_input_size = int(input_size/8)

    vgg_encoder_model = vgg_encoder(input_shape=[input_size,input_size,3])
    vgg_encoder_model.load_weights(DEFAULT_VGG_ENCODER_H5)
    vgg_combine_decoder = combine_and_decode_model(input_shape=[decoder_input_size,decoder_input_size,512],
                                                   model="vgg",
                                                   include_post_process=False)
    vgg_combine_decoder.load_weights(DEFAULT_VGG_DECODER_H5, by_name=False)
    model = build_model(vgg_combine_decoder)
    model.load_weights("mobile_decoder.h5", by_name=True)
     
    c_fnames = glob.glob("input/content/chicago.jpg")
    s_fnames = glob.glob("input/style/asheville.jpg")
    print(len(c_fnames), len(s_fnames))
      
    # c_fnames, s_fnames, batch_size, shuffle, encoder_model, combine_decoder_model
    train_generator = CombineBatchGenerator(c_fnames,
                                            s_fnames,
                                            batch_size=min(args.batch_size, len(s_fnames), len(s_fnames)),
                                            shuffle=True,
                                            encoder_model=vgg_encoder_model,
                                            combine_decoder_model=vgg_combine_decoder,
                                            input_size=input_size)
       
    # 2. create loss function
    if USE_TF_KERAS:
        opt = tf.keras.optimizers.Adam(lr=args.learning_rate)
    else:
        opt = keras.optimizers.Adam(lr=args.learning_rate)
    model.compile(loss=loss_func,
                  optimizer=opt)
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator),
                        callbacks=create_callbacks(saved_weights_name="mobile_decoder.h5"),
                        validation_data  = train_generator,
                        validation_steps = len(train_generator),
                        epochs=1000)

