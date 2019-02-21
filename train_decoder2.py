
# import modules
import numpy as np
import os
import glob
import tensorflow as tf
import keras
import argparse

np.random.seed(1337)
from adain import MODEL_ROOT, USE_TF_KERAS, DEFAULT_NEW_BLOCK
from adain.encoder import vgg_encoder, extract_feature_model
from adain.decoder import combine_and_decode_model
from adain.generator import CombineBatchGenerator, create_callbacks


DEFAULT_IMG_ROOT = os.path.join("experiments", "imgs")
DEFAULT_BATCH_SIZE = 4

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_INPUT_SIZE = 256


argparser = argparse.ArgumentParser(description='Train mobile decoder')
argparser.add_argument('-l',
                       '--learning_rate',
                       default=DEFAULT_LEARNING_RATE,
                       type=float,
                       help='learning rate')
argparser.add_argument('-s',
                       '--size',
                       default=DEFAULT_INPUT_SIZE,
                       help='images directory')
argparser.add_argument('-n',
                       '--new_block',
                       default=DEFAULT_NEW_BLOCK,
                       help='images directory')

argparser.add_argument('-i',
                       '--image_root',
                       default=DEFAULT_IMG_ROOT,
                       help='images directory')
argparser.add_argument('-b',
                       '--batch_size',
                       default=DEFAULT_BATCH_SIZE,
                       type=int,
                       help='batch size')


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


def loss_func(y_true, y_pred):
    # 1. activate prediction & truth tensor
    # style_loss = tf.losses.mean_squared_error(y_true, y_pred)
    # print(y_true.shape, y_pred.shape)
    loss = tf.losses.mean_squared_error(y_true, y_pred)
    return loss


def create_models(input_size, num_new_blocks):
    
    decoder_input_size = int(input_size/8)
    vgg_encoder_model = vgg_encoder(input_size)
    vgg_combine_decoder = combine_and_decode_model(feature_size=decoder_input_size,
                                                   include_post_process=False)
    
    model = build_mobile_combine_decoder(feature_size=decoder_input_size,
                                         num_new_blocks=num_new_blocks,
                                         include_post_process=True)
    
    x = model.layers[-1].output
    feature_model = extract_feature_model()
    x = feature_model(x)
    model = Model(model.inputs, x, name='mobile_decoder_encoder')
    model.layers[-1].trainable = False
    model.summary()
    # model.load_weights("mobile_decoder.h5", by_name=True)
    return vgg_encoder_model, vgg_combine_decoder, model


from adain.transfer_decoder import build_mobile_combine_decoder
if __name__ == '__main__':
    args = argparser.parse_args()
    vgg_encoder_model, vgg_combine_decoder, model = create_models(args.size, args.new_block)
    # vgg_combine_decoder.summary()
     
    c_fnames = glob.glob("input/content/chicago.jpg")
    s_fnames = glob.glob("input/style/asheville.jpg")
    
#     c_fnames = glob.glob("input/content/*.*")
#     s_fnames = glob.glob("input/style/*.*")
    print(len(c_fnames), len(s_fnames))
       
    # c_fnames, s_fnames, batch_size, shuffle, encoder_model, combine_decoder_model
    train_generator = CombineBatchGenerator(c_fnames,
                                            s_fnames,
                                            batch_size=min(args.batch_size, len(s_fnames), len(s_fnames)),
                                            shuffle=True,
                                            encoder_model=vgg_encoder_model,
                                            combine_decoder_model=vgg_combine_decoder,
                                            input_size=args.size)
    
#     xs, ys = train_generator[0]
#     print(xs[0].shape, xs[1].shape, ys.shape)
        
    # 2. create loss function
    model.compile(loss=loss_func,
                  optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate))
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator),
                        callbacks=create_callbacks(saved_weights_name="mobile_decoder.h5"),
                        validation_data  = train_generator,
                        validation_steps = len(train_generator),
                        epochs=1000)

