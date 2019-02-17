# -*- coding: utf-8 -*-

import tensorflow as tf
import subprocess
from adain.graph import freeze_session, load_graph_from_pb
from adain import PKG_ROOT

if __name__ == '__main__':
    from adain.encoder import vgg_encoder, mobile_encoder
    from adain.transfer_decoder import build_mobile_combine_decoder
    tf.keras.backend.set_learning_phase(0) # this line most important

    ##########################################################################
    pb_fname = "decoder.pb"
    output_pb_fname = "mobile_decoder_opt.pb"
    input_node = "input_c,input_s"
    output_node = "output_1/mul"
    ##########################################################################

    model = build_mobile_combine_decoder(include_post_process=True)
    model.load_weights(PKG_ROOT + "/models/h5/mobile_decoder.h5")
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

