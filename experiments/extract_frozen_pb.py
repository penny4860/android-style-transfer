
import tensorflow as tf
from adain.graph import freeze_session
from adain import PKG_ROOT

if __name__ == '__main__':
    from adain.transfer_decoder import build_mobile_combine_decoder
    # encoder_model = vgg_encoder()
    model = build_mobile_combine_decoder()
    model.load_weights(PKG_ROOT + "/models/h5/mobile_decoder.h5")
    model.summary()
        
    # 1. to frozen pb
    K = tf.keras.backend
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "models", "decoder.pb", as_text=False)
    # input_c,input_s  / output/mul
    for t in model.inputs + model.outputs:
        print("op name: {}, shape: {}".format(t.op.name, t.shape))

    # python -m tensorflow.python.tools.optimize_for_inference --input encoder.pb --output mobile_encoder_opt.pb --input_names=input --output_names=output/Relu

