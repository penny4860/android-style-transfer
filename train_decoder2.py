
# import modules
import numpy as np
# import tensorflow as tf
import os
import glob
import keras
import argparse

np.random.seed(1337)
from adain import MODEL_ROOT
from adain.encoder import vgg_encoder
from adain.decoder import combine_and_decode_model
from adain.generator import CombineBatchGenerator, create_callbacks


DEFAULT_IMG_ROOT = os.path.join("experiments", "imgs")
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 0.01
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


if __name__ == '__main__':
    args = argparser.parse_args()
    
    input_size = 512
    decoder_input_size = int(input_size/8)

    vgg_encoder_model = vgg_encoder(input_shape=[input_size,input_size,3])
    vgg_encoder_model.load_weights(DEFAULT_VGG_ENCODER_H5)
    vgg_combine_decoder = combine_and_decode_model(input_shape=[decoder_input_size,decoder_input_size,512],
                                                   model="vgg",
                                                   include_post_process=False)
    vgg_combine_decoder.load_weights(DEFAULT_VGG_DECODER_H5, by_name=False)
    model = combine_and_decode_model(input_shape=[decoder_input_size,decoder_input_size,512],
                                     model="mobile",
                                     include_post_process=False,
                                     use_bn=False)
    if args.weights_init:
        model.load_weights(args.weights_init, by_name=True)
    
    c_fnames = glob.glob("input/content/*.*")
    s_fnames = glob.glob("input/style/*.*")
    print(len(c_fnames), len(s_fnames))
    
    # c_fnames, s_fnames, batch_size, shuffle, encoder_model, combine_decoder_model
    train_generator = CombineBatchGenerator(c_fnames,
                                            s_fnames,
                                            batch_size=min(args.batch_size, len(s_fnames), len(s_fnames)),
                                            shuffle=True,
                                            encoder_model=vgg_encoder_model,
                                            combine_decoder_model=vgg_combine_decoder,
                                            input_size=input_size)
#     import matplotlib.pyplot as plt
#     for j in range(5):
#         xs, ys = train_generator[j]
#         for i in range(4):
#             plt.imshow(ys[i].astype(np.uint8))
#             plt.show()
    
#     # valid_generator = BatchGenerator(fnames[160:], batch_size=4, shuffle=False)
     
    # 2. create loss function
    model.compile(loss="mean_squared_error",
                  optimizer=keras.optimizers.Adam(lr=args.learning_rate))
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator),
                        callbacks=create_callbacks(saved_weights_name="mobile_decoder.h5"),
                        validation_data  = train_generator,
                        validation_steps = len(train_generator),
                        epochs=1000)

