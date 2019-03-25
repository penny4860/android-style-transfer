# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from adain import PROJECT_ROOT

# if USE_TF_KERAS:
Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
BatchNormalization = tf.keras.layers.BatchNormalization
Activateion = tf.keras.layers.Activation
MaxPooling2D = tf.keras.layers.MaxPooling2D
Layer = tf.keras.layers.Layer
Model = tf.keras.models.Model
VGG19 = tf.keras.applications.vgg19.VGG19


class SpatialReflectionPadding(Layer):

    def __init__(self, **kwargs):
        super(SpatialReflectionPadding, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]+2, input_shape[2]+2, input_shape[3])

    def call(self, x):
        return tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")


def vgg_encoder(input_size=256):
    
    def _build_model(input_shape):
        x = Input(shape=input_shape, name="input")
        img_input = x
    
        # Block 1
        # x = VggPreprocess()(x)
        x = Conv2D(3, (1, 1), activation=None, padding='same', name='block1_conv0')(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
           
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
           
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)


#         x = SpatialReflectionPadding()(x)
#         x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv2')(x)
#         x = SpatialReflectionPadding()(x)
#         x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv3')(x)
#         x = SpatialReflectionPadding()(x)
#         x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv4')(x)
#         x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#         
#         # Block 4
#         x = SpatialReflectionPadding()(x)
#         x = Conv2D(512, (3, 3), activation='relu', padding='valid', name="output")(x)
        model = Model(img_input, x, name='vgg19')
        return model

    input_shape=[input_size,input_size,3]
    model = _build_model(input_shape)
    
    fname = os.path.join(PROJECT_ROOT, "linear", "models", "h5", "vgg_31.h5")
    model.load_weights(fname, by_name=True)
    return model

import subprocess
from adain.graph import freeze_session, load_graph_from_pb
if __name__ == '__main__':
#     tf.keras.backend.set_learning_phase(0) # this line most important
# 
    ##########################################################################
    pb_fname = "vgg_31.pb"
    output_pb_fname = "vgg_31_opt.pb"
    input_node = "input"
    output_node = "block3_conv1/Relu"
    ##########################################################################
 
    model = vgg_encoder(input_size=256)
    model.summary()
         
    # 1. to frozen pb
    K = tf.keras.backend
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "models", pb_fname, as_text=False)
    # input_c,input_s  / output/mul
    for t in model.inputs + model.outputs:
        print("op name: {}, shape: {}".format(t.op.name, t.shape))
   
    cmd = 'python -m tensorflow.python.tools.optimize_for_inference \
            --input models/{} \
            --output {} \
            --input_names={} \
            --output_names={}'.format(pb_fname, output_pb_fname, input_node, output_node)
    subprocess.call(cmd, shell=True)
    
    sess = load_graph_from_pb(output_pb_fname)
    print(sess)


    graph_def_file = "models/" + pb_fname
    graph_def_file = output_pb_fname
    
    input_arrays = ["input"]
    output_arrays = [output_node]
    
    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                          input_arrays,
                                                          output_arrays,
                                                          input_shapes={"input":[1,256,256,3]})
    tflite_model = converter.convert()
    open("vgg_31.tflite", "wb").write(tflite_model)
     
    
