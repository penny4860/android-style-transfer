package compenny4860.github.tensorflowliteexample;

import android.app.Activity;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class Decoder {
    private TensorFlowInferenceInterface decoderInterface;
    private static final String MODEL_DECODER_FILE = "decoder_31_opt.pb";

    private static final String INPUT_C_NODE = "input";
    private static final String OUTPUT_NODE = "post_preprocess/mul";

    Decoder(Activity activity) {
        decoderInterface = new TensorFlowInferenceInterface(activity.getAssets(), MODEL_DECODER_FILE);
    }

    public void run(float stylized_img[], float csFeatureValues[], int imgSize) {
        decoderInterface.feed(INPUT_C_NODE, csFeatureValues,
                1, imgSize, imgSize, 256);
        decoderInterface.run(new String[] {OUTPUT_NODE}, false);
        decoderInterface.fetch(OUTPUT_NODE, stylized_img);
    }
}