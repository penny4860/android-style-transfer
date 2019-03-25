# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

from adain.utils import preprocess, plot
from adain import MODEL_ROOT, PROJECT_ROOT


# LINEAR_H5_ROOT = os.path.join(PROJECT_ROOT, "linear/models/h5")
# LINEAR_ENCODER_H5 = os.path.join(LINEAR_H5_ROOT, "vgg_31.h5")
# LINEAR_DECODER_H5 = os.path.join(LINEAR_H5_ROOT, "vgg_decoder_31.h5")
# LINEAR_MAT_H5 = os.path.join(LINEAR_H5_ROOT, "mat.h5")

DEFAULT_ENCODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_encoder.h5")
DEFAULT_DECODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_decoder.h5")

content_fname="../input/content/chicago.jpg"
style_fname="../input/style/asheville.jpg"
alpha = 1.0
img_size = 416

if __name__ == '__main__':
    
    from linear.encoder import vgg_encoder
    from linear.decoder import vgg_decoder
    from linear.mat import build_model as mat_model
    
    encoder = vgg_encoder(img_size)
    decoder = vgg_decoder(int(img_size)/4)
    mat = mat_model(input_shape=[int(img_size/4),
                                 int(img_size/4),
                                 256])
    
    # 1. contents / style images
    c_img = cv2.imread(content_fname)[:,:,::-1]
    s_img = cv2.imread(style_fname)[:,:,::-1]

    # 2. load input imgs
    c_img_prep = preprocess(c_img, (img_size,img_size)) / 255.0
    s_img_prep = preprocess(s_img, (img_size,img_size)) / 255.0
#     
#     # 3. encoding
#     from adain.encoder import vgg_encoder
#     from adain.decoder import combine_and_decode_model
#     encoder = vgg_encoder(img_size)
#     decoder = combine_and_decode_model(img_size/8)
#     encoder.load_weights(DEFAULT_ENCODER_H5)
#     decoder.load_weights(DEFAULT_DECODER_H5)
# 

    print(c_img_prep.shape, s_img_prep.shape)
    c_features = encoder.predict(c_img_prep)
    s_features = encoder.predict(s_img_prep)

    cs_features = mat.predict([c_features, s_features])
 
    stylized_imgs = decoder.predict(cs_features)
    stylized_imgs[stylized_imgs>1] = 1
    stylized_imgs[stylized_imgs<0] = 0
    stylized_imgs *= 255
    stylized_img = stylized_imgs[0].astype(np.uint8)
    
    # 5. plot
    plot([c_img, s_img, stylized_img])
