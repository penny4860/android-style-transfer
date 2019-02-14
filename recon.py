# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

from adain.utils import preprocess, plot
from adain import MODEL_ROOT, PROJECT_ROOT

DEFAULT_ENCODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_encoder.h5")
DEFAULT_DECODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_decoder.h5")

content_fname="input/content/chicago.jpg"
style_fname="input/style/asheville.jpg"
alpha = 1.0


if __name__ == '__main__':
    
    encoder_input = 512
    decoder_input = int(encoder_input/8)
    
    # 1. contents / style images
    c_img = cv2.imread(content_fname)[:,:,::-1]
    s_img = cv2.imread(style_fname)[:,:,::-1]

    # 2. load input imgs
    c_img_prep = preprocess(c_img, (encoder_input,encoder_input))
    s_img_prep = preprocess(s_img, (encoder_input,encoder_input))
    
    # 3. encoding
    from adain.encoder import vgg_encoder
    from adain.decoder import combine_and_decode_model
    from train_decoder2 import build_model
    vgg_encoder = vgg_encoder(input_shape=[encoder_input,encoder_input,3])
    vgg_encoder.load_weights(DEFAULT_ENCODER_H5)
    mobile_decoder = build_model(combine_and_decode_model(input_shape=[decoder_input,decoder_input,512], model="vgg"))
    mobile_decoder.summary()
    print(len(mobile_decoder.layers))
    mobile_decoder.load_weights("mobile_decoder.h5")

    c_features = vgg_encoder.predict(c_img_prep)
    s_features = vgg_encoder.predict(s_img_prep)

    stylized_imgs = mobile_decoder.predict([c_features, s_features])
    stylized_img = stylized_imgs[0]
    stylized_img[stylized_img > 1] = 1
    stylized_img[stylized_img < 0] = 0
    stylized_img *= 255
    stylized_img = stylized_img.astype(np.uint8)

    # 5. plot
    plot([c_img, s_img, stylized_img])
