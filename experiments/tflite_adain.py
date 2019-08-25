# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import tensorflow as tf

from adain.utils import preprocess, plot
from adain import MODEL_ROOT

DEFAULT_ENCODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_encoder.h5")
DEFAULT_DECODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_decoder.h5")

content_fname="../input/content/chicago.jpg"
style_fname="../input/style/asheville.jpg"
img_size = 416


def extract_feat_tflite(model, input_data):
    # Convert to TF Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test model on random input data.
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def decode(model, s_features, c_features):
    # Convert to TF Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
     
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
     
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    print(input_details[0]["name"])
    print(input_details[1]["name"])
    
    # Test model on random input data.
    if input_details[0]["name"] == "input_s":
        interpreter.set_tensor(input_details[0]['index'], s_features)
        interpreter.set_tensor(input_details[1]['index'], c_features)
    else:
        interpreter.set_tensor(input_details[0]['index'], c_features)
        interpreter.set_tensor(input_details[1]['index'], s_features)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    stylized_imgs = interpreter.get_tensor(output_details[0]['index'])
    return stylized_imgs


if __name__ == '__main__':
    
    # 1. contents / style images
    c_img = cv2.imread(content_fname)[:,:,::-1]
    s_img = cv2.imread(style_fname)[:,:,::-1]

    # 2. load input imgs
    c_img_prep = preprocess(c_img, (img_size,img_size))
    s_img_prep = preprocess(s_img, (img_size,img_size))
    
    # 3. encoding
    from adain.encoder import vgg_encoder
    model = vgg_encoder(img_size)
    model.load_weights(DEFAULT_ENCODER_H5)
    
    c_features = extract_feat_tflite(model, c_img_prep)
    s_features = extract_feat_tflite(model, s_img_prep)
    print(c_features.shape, s_features.shape)
    
 
    from adain.decoder import combine_and_decode_model
    decoder = combine_and_decode_model(img_size/8)
    decoder.load_weights(DEFAULT_DECODER_H5)
    stylized_imgs = decode(decoder, s_features, c_features)
     
    stylized_img = stylized_imgs[0].astype(np.uint8)
    
    # 5. plot
    plot([c_img, s_img, stylized_img])
