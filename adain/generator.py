# -*- coding: utf-8 -*-

import cv2
import numpy as np

from adain import USE_TF_KERAS

if USE_TF_KERAS:
    import tensorflow as tf
    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
    TensorBoard = tf.keras.callbacks.TensorBoard
    ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
    Sequence = tf.keras.utils.Sequence
else:
    import keras
    ModelCheckpoint = keras.callbacks.ModelCheckpoint
    TensorBoard = keras.callbacks.TensorBoard
    ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
    Sequence = keras.utils.Sequence


def create_callbacks(saved_weights_name="mobile_encoder.h5"):
    checkpoint = ModelCheckpoint(saved_weights_name, 
                                                    monitor='val_loss', 
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    mode='min', 
                                                    period=1)
    
#     tensorboard_callback = TensorBoard(
#             histogram_freq         = 0,
#             write_graph            = True,
#             write_grads            = False,
#             write_images           = False,
#             embeddings_freq        = 0,
#             embeddings_layer_names = None,
#             embeddings_metadata    = None
#         )
    callbacks = [checkpoint]
#     callbacks.append(tensorboard_callback)
    callbacks.append(ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.5,
        patience   = 10,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 1e-5
    ))
    return callbacks




class BatchGenerator(Sequence):
    def __init__(self, fnames, batch_size, shuffle, truth_model, input_size=256):
        self.fnames = fnames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        self.truth_model = truth_model        
        self.truth_model.predict(np.zeros((1,input_size,input_size,3)))
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.fnames) /self.batch_size)

    def __getitem__(self, idx):
        """
        # Args
            idx : batch index
        """
        batch_fnames = self.fnames[idx*self.batch_size: (idx+1)*self.batch_size]
        xs = [cv2.imread(fname)[:,:,::-1] for fname in batch_fnames]
        xs = np.array([cv2.resize(img, (self.input_size,self.input_size)) for img in xs])
        ys = self.truth_model.predict(xs)
        return xs, ys

    def on_epoch_end(self):
        np.random.shuffle(self.fnames)


class DecodeBatchGenerator(Sequence):
    def __init__(self, fnames, batch_size, shuffle, encoder_model, decoder_model, input_size=256):
        self.fnames = fnames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        
        self.vgg_encoder = encoder_model
        self.vgg_encoder.predict(np.zeros((1,input_size,input_size,3)))
        self.vgg_decoder = decoder_model
        self.vgg_decoder.predict(np.zeros((1,int(input_size/8),int(input_size/8),512)))
        
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.fnames) /self.batch_size)

    def __getitem__(self, idx):
        """
        # Args
            idx : batch index
        """
        batch_fnames = self.fnames[idx*self.batch_size: (idx+1)*self.batch_size]
        imgs = [cv2.imread(fname)[:,:,::-1] for fname in batch_fnames]
        imgs = np.array([cv2.resize(img, (self.input_size,self.input_size)) for img in imgs])
        
        xs = self.vgg_encoder.predict(imgs)
        ys = self.vgg_decoder.predict(xs)
        return xs, ys

    def on_epoch_end(self):
        np.random.shuffle(self.fnames)


class CombineBatchGenerator(Sequence):
    def __init__(self, c_fnames, s_fnames, batch_size, shuffle, encoder_model, combine_decoder_model, input_size=256):
        self.c_fnames = c_fnames
        self.s_fnames = s_fnames
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        
        self.vgg_encoder = encoder_model
        self.vgg_encoder.predict(np.zeros((1,input_size,input_size,3)))
        self.vgg_combine_decoder = combine_decoder_model
        self.vgg_combine_decoder.predict([np.zeros((1,int(input_size/8),int(input_size/8),512)),
                                          np.zeros((1,int(input_size/8),int(input_size/8),512))])
        
        self.on_epoch_end()

    def __len__(self):
        n_files = min(len(self.c_fnames), len(self.s_fnames))
        return int(n_files / self.batch_size)

    def __getitem__(self, idx):
        """
        # Args
            idx : batch index
        """
        batch_c_fnames = self.c_fnames[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_s_fnames = self.s_fnames[idx*self.batch_size: (idx+1)*self.batch_size]
        
        c_imgs = [cv2.imread(fname)[:,:,::-1] for fname in batch_c_fnames]
        c_imgs = np.array([cv2.resize(img, (self.input_size,self.input_size)) for img in c_imgs])

        s_imgs = [cv2.imread(fname)[:,:,::-1] for fname in batch_s_fnames]
        s_imgs = np.array([cv2.resize(img, (self.input_size,self.input_size)) for img in s_imgs])
        
        c_features = self.vgg_encoder.predict(c_imgs)
        s_features = self.vgg_encoder.predict(s_imgs)
        
        xs = [c_features, s_features]
        ys = self.vgg_combine_decoder.predict(xs)
        return xs, ys

    def on_epoch_end(self):
        np.random.shuffle(self.c_fnames)
        np.random.shuffle(self.s_fnames)


