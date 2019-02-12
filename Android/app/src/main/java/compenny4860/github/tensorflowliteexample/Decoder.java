package compenny4860.github.tensorflowliteexample;

import android.app.Activity;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class Decoder {
    private TensorFlowInferenceInterface decoderInterface;
    private static final String MODEL_DECODER_FILE = "decoder_opt.pb";

    private static final String INPUT_C_NODE = "input_c";
    private static final String INPUT_S_NODE = "input_s";
    private static final String OUTPUT_NODE = "output/mul";

    Decoder(Activity activity) {
        decoderInterface = new TensorFlowInferenceInterface(activity.getAssets(), MODEL_DECODER_FILE);
    }

    public void run(float stylized_img[], float contentFeatureValues[], float styleFeatureValues[], int imgSize) {
        decoderInterface.feed(INPUT_C_NODE, contentFeatureValues,
                1, imgSize, imgSize, 512);
        decoderInterface.feed(INPUT_S_NODE, styleFeatureValues,
                1, imgSize, imgSize, 512);
        decoderInterface.run(new String[] {OUTPUT_NODE}, false);
        decoderInterface.fetch(OUTPUT_NODE, stylized_img);
    }
}