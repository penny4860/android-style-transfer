# -*- coding: utf-8 -*-

import tensorflow as tf
import subprocess
from adain.graph import freeze_session, load_graph_from_pb
from adain import PKG_ROOT, PROJECT_ROOT

if __name__ == '__main__':
    from linear.encoder import vgg_encoder, mobile_encoder
    tf.keras.backend.set_learning_phase(0) # this line most important

    ##########################################################################
    pb_fname = "mobile_31.pb"
    output_pb_fname = "mobile_31_opt.pb"
    input_node = "input"
    output_node = "block3_conv1/Relu"
    ##########################################################################

    model = mobile_encoder()
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

