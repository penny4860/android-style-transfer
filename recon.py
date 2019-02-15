# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

from adain.utils import preprocess, plot
from adain import MODEL_ROOT
from adain.encoder import vgg_encoder
from adain.decoder import combine_and_decode_model
from adain.transfer_decoder import build_mobile_combine_decoder


DEFAULT_DECODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_decoder.h5")

content_fname="input/content/chicago.jpg"
style_fname="input/style/asheville.jpg"
alpha = 1.0


def postprocess(image):
    image[image > 1] = 1
    image[image < 0] = 0
    image *= 255
    image = image.astype(np.uint8)
    return image

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
    encoder = vgg_encoder(input_size=encoder_input)
    mobile_decoder = build_mobile_combine_decoder(decoder_input, 2)
    mobile_decoder.load_weights("mobile_decoder.h5")

    c_features = encoder.predict(c_img_prep)
    s_features = encoder.predict(s_img_prep)

    stylized_imgs = mobile_decoder.predict([c_features, s_features])
    img = postprocess(stylized_imgs[0])

    # 5. plot
    plot([c_img, s_img, img])
