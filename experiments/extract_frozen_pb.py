# -*- coding: utf-8 -*-

import tensorflow as tf
import subprocess
from adain.graph import freeze_session, load_graph_from_pb
from adain import PKG_ROOT

if __name__ == '__main__':
    from adain.encoder import vgg_encoder, mobile_encoder
    from adain.transfer_decoder import build_mobile_combine_decoder

    ##########################################################################
    pb_fname = "mobile_decoder.pb"
    ##########################################################################

#     model = build_mobile_combine_decoder()
#     model.load_weights(PKG_ROOT + "/models/h5/mobile_decoder.h5")
#     model.summary()
#         
#     # 1. to frozen pb
#     K = tf.keras.backend
#     frozen_graph = freeze_session(K.get_session(),
#                                   output_names=[out.op.name for out in model.outputs])
#     tf.train.write_graph(frozen_graph, "models", pb_fname, as_text=False)
#     # input_c,input_s  / output/mul
#     for t in model.inputs + model.outputs:
#         print("op name: {}, shape: {}".format(t.op.name, t.shape))

    # graph optimize : tf 1.4.0
#     cmd = 'python -m tensorflow.python.tools.optimize_for_inference \
#             --input models/{} \
#             --output {} \
#             --input_names={} \
#             --output_names={}'.format(pb_fname, output_pb_fname, input_node, output_node)
#     subprocess.call(cmd, shell=True)
# 
    sess = load_graph_from_pb("models/opt.pb")
    print(sess)


