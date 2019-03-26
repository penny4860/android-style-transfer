package compenny4860.github.tensorflowliteexample;

import android.app.Activity;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class Mat {
    private TensorFlowInferenceInterface matInterface;
    private static final String MODEL_DECODER_FILE = "mat_31_opt.pb";

    private static final String INPUT_C_NODE = "input_c";
    private static final String INPUT_S_NODE = "input_s";
    private static final String OUTPUT_NODE = "mean_add/add";

    Mat(Activity activity) {
        matInterface = new TensorFlowInferenceInterface(activity.getAssets(), MODEL_DECODER_FILE);
    }
    public void run(float csFeatureValues[], float contentFeatureValues[], float styleFeatureValues[], int imgSize) {
        matInterface.feed(INPUT_C_NODE, contentFeatureValues,
                1, imgSize, imgSize, 256);
        matInterface.feed(INPUT_S_NODE, styleFeatureValues,
                1, imgSize, imgSize, 256);
        matInterface.run(new String[] {OUTPUT_NODE}, false);
        matInterface.fetch(OUTPUT_NODE, csFeatureValues);
    }
}



