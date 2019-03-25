# -*- coding: utf-8 -*-

import numpy as np
import cv2

from adain.utils import preprocess, plot

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
    c_img_prep = preprocess(c_img, (img_size,img_size))
    s_img_prep = preprocess(s_img, (img_size,img_size))
 
    # 3. encoding
    c_features = encoder.predict(c_img_prep)
    s_features = encoder.predict(s_img_prep)
 
    # 4. mix
    cs_features = mat.predict([c_features, s_features])
 
    # 5. decode
    stylized_imgs = decoder.predict(cs_features)
 
    # 6. post process
    stylized_imgs[stylized_imgs>1] = 1
    stylized_imgs[stylized_imgs<0] = 0
    stylized_imgs *= 255
    stylized_img = stylized_imgs[0].astype(np.uint8)
     
    # 7. plot
    plot([c_img, s_img, stylized_img])
