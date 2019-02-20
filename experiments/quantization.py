# -*- coding: utf-8 -*-

# References : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
from tensorflow.tools.graph_transforms import TransformGraph
from adain.graph import load_graph_def_from_pb

import os
import tensorflow as tf
from adain import PKG_ROOT

INPUT_PB_FNAME = os.path.join(PKG_ROOT, "models", "decoder_opt.pb")
OUTPUT_PB_FNAME = os.path.join(PKG_ROOT, "models", "decoder_opt_quantized.pb")

INPUT_NAMES = ["input_c", "input_s"]
OUTPUT_NAMES = ["output/mul"]
TRANSFORMS = ["fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms quantize_weights"]


if __name__ == "__main__":
    
    #1. load graph
    print("Input pb filename: {}".format(INPUT_PB_FNAME))
    graph_def = load_graph_def_from_pb(INPUT_PB_FNAME)
 
    # 2. define transform
    transformed_graph_def = TransformGraph(graph_def,
                                           INPUT_NAMES,
                                           OUTPUT_NAMES,
                                           TRANSFORMS)
     
    # 3. run
    tf.train.write_graph(transformed_graph_def,
                         os.path.dirname(OUTPUT_PB_FNAME),
                         os.path.basename(OUTPUT_PB_FNAME),
                         as_text=False)
    print("Output pb filename: {}".format(OUTPUT_PB_FNAME))
    

