
import tensorflow as tf
import subprocess
from adain.graph import freeze_session
from adain import PKG_ROOT

if __name__ == '__main__':
    from adain.encoder import vgg_encoder, mobile_encoder
    from adain.transfer_decoder import build_mobile_combine_decoder

    ##########################################################################
    pb_fname = "encoder.pb"
    output_pb_fname = "encoder_opt.pb"
    input_node = "input"
    output_node = "output/Relu"
    ##########################################################################

    model = mobile_encoder()
    model.load_weights(PKG_ROOT + "/models/h5/mobile_encoder.h5")
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


