
# import modules
import numpy as np
import keras
import os
import glob

np.random.seed(1337)
from adain.generator import BatchGenerator, create_callbacks
from linear.encoder import vgg_encoder, mobile_encoder

DEFAULT_IMG_ROOT = os.path.join("experiments", "imgs")
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_INIT_WEIGHTS = os.path.join("experiments", "mobile_encoder.h5")
DEFAULT_VGG_WEIGHTS = os.path.join("adain", "models", "vgg_encoder.h5")

import argparse
argparser = argparse.ArgumentParser(description='Train mobile encoder')

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
    
    input_size = 256
    teacher_model = vgg_encoder(input_size)
    student_model = mobile_encoder(input_size)
    
    fnames = glob.glob(args.image_root + "/*.jpg")
    print("{}-files to train".format(len(fnames)))
    train_generator = BatchGenerator(fnames,
                                     batch_size=min(args.batch_size, len(fnames)),
                                     shuffle=False,
                                     truth_model=teacher_model,
                                     input_size=input_size)
    # valid_generator = BatchGenerator(fnames[160:], batch_size=4, shuffle=False)
    
    # 2. create loss function
    student_model.compile(loss="mean_squared_error",
                  optimizer=keras.optimizers.Adam(lr=args.learning_rate))
    student_model.fit_generator(train_generator,
                                steps_per_epoch=len(train_generator),
                                callbacks=create_callbacks(),
                                validation_data  = train_generator,
                                validation_steps = len(train_generator),
                                epochs=1000)
