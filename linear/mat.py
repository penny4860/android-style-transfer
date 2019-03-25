
import tensorflow as tf
Input = tf.keras.layers.Input
Layer = tf.keras.layers.Layer
Model = tf.keras.models.Model
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
Add = tf.keras.layers.Add

class MeanSub(Layer):
    def __init__(self, **kwargs):
        super(MeanSub, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        x = x - tf.reduce_mean(x, [1,2], keep_dims=True)
        return x

class Flat(Layer):
    def __init__(self, **kwargs):
        super(Flat, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return [input_shape[0] * input_shape[1], input_shape[2]]

    def call(self, x):
        _, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, [-1, h*w, c])
        # (4096, 32)
        x = tf.matmul(tf.transpose(x, [0,2,1]), x) / (h*w)
        _, c1, c2 = x.get_shape().as_list()
        x = tf.reshape(x, [-1, c1*c2])
        return x

class UnFlat(Layer):
    def __init__(self, mat_size=32, **kwargs):
        super(UnFlat, self).__init__(**kwargs)
        self.mat_size = mat_size

    def compute_output_shape(self, input_shape):
        assert self.mat_size == input_shape[0]
        assert self.mat_size == input_shape[1]
        return [self.mat_size * self.mat_size]

    def call(self, x):
        x = tf.reshape(x, [-1, self.mat_size, self.mat_size])
        return x

class TransMatrix(Layer):
    def __init__(self, **kwargs):
        super(TransMatrix, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, xs):
        x1, x2 = xs
        x = tf.matmul(x1, x2)
        return x

class TransFeat(Layer):
    def __init__(self, **kwargs):
        super(TransFeat, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def call(self, xs):
        transmatrix, compress_content = xs
        _, h, w, c = compress_content.get_shape().as_list()

        compress_content = tf.reshape(compress_content, [-1, h*w, c])
        compress_content = tf.transpose(compress_content, [0, 2, 1])
        x = tf.matmul(transmatrix, compress_content)
        x = tf.reshape(x, [-1, c, h, w])
        x = tf.transpose(x, [0, 2, 3, 1])
        return x


class MeanAdd(Layer):
    def __init__(self, **kwargs):
        super(MeanAdd, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, xs):
        x, input_s = xs
        style_mean = tf.reduce_mean(input_s, [1,2], keep_dims=True)
        return x + style_mean


def build_model(input_shape=[64,64,256]):
    input_c = Input(shape=input_shape, name="input_c")
    input_s = Input(shape=input_shape, name="input_s")
    
    # Block 1
    xc = MeanSub()(input_c)
    contents = xc
    xc = Conv2D(128, (3,3), activation='relu', padding='same', name="c1")(xc)
    xc = Conv2D(64, (3,3), activation='relu', padding='same', name="c2")(xc)
    xc = Conv2D(32, (3,3), activation=None, padding='same', name="c3")(xc)
    xc = Flat()(xc)
    xc = Dense(1024, name="c4")(xc)
    xc = UnFlat()(xc)

    xs = MeanSub()(input_s)
    xs = Conv2D(128, (3,3), activation='relu', padding='same', name="s1")(xs)
    xs = Conv2D(64, (3,3), activation='relu', padding='same', name="s2")(xs)
    xs = Conv2D(32, (3,3), activation=None, padding='same', name="s3")(xs)
    xs = Flat()(xs)
    xs = Dense(1024, name="s4")(xs)
    xs = UnFlat()(xs)
    
    x = TransMatrix()([xs, xc])
    
    compress = Conv2D(32, (1, 1), activation=None, padding='valid', name='compress')(contents)
    x = TransFeat()([x, compress])
    x = Conv2D(256, (1,1), activation=None, padding='same', name="unzip")(x)
    x = MeanAdd()([x, input_s])
    
    model = Model([input_c, input_s], x, name='mat')
    
    import os
    dname = os.path.dirname(__file__)
    model.load_weights(os.path.join(dname, "mat.h5"))
    return model


import subprocess
from adain.graph import freeze_session, load_graph_from_pb
from adain import PKG_ROOT, PROJECT_ROOT
if __name__ == '__main__':
#     tf.keras.backend.set_learning_phase(0) # this line most important
# 
    ##########################################################################
    pb_fname = "mat_31.pb"
    output_pb_fname = "mat_31_opt.pb"
    input_node = "input_c,input_s"
    output_node = "mean_add/add"
    ##########################################################################
 
    model = build_model()
    model.summary()
         
#     # 1. to frozen pb
#     K = tf.keras.backend
#     frozen_graph = freeze_session(K.get_session(),
#                                   output_names=[out.op.name for out in model.outputs])
#     tf.train.write_graph(frozen_graph, "models", pb_fname, as_text=False)
#     # input_c,input_s  / output/mul
#     for t in model.inputs + model.outputs:
#         print("op name: {}, shape: {}".format(t.op.name, t.shape))
#       
#     cmd = 'python -m tensorflow.python.tools.optimize_for_inference \
#             --input models/{} \
#             --output {} \
#             --input_names={} \
#             --output_names={}'.format(pb_fname, output_pb_fname, input_node, output_node)
#     subprocess.call(cmd, shell=True)
#        
#     sess = load_graph_from_pb(output_pb_fname)
#     print(sess)
   
  
    graph_def_file = "models/" + pb_fname
    graph_def_file = output_pb_fname
      
    input_arrays = ["input_c", "input_s"]
    output_arrays = [output_node]
      
    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                          input_arrays,
                                                          output_arrays,
                                                          input_shapes={"input_c":[1,64,64,256], "input_s":[1,64,64,256]})
    tflite_model = converter.convert()
    open("mat_31.tflite", "wb").write(tflite_model)

