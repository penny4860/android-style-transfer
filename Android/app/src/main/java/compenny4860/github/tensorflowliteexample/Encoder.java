package compenny4860.github.tensorflowliteexample;

import android.app.Activity;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class Encoder {
    private TensorFlowInferenceInterface encoderInterface;
    private static final String MODEL_ENCODER_FILE = "vgg_31_opt.pb";
    private static final String INPUT_NODE = "input";
    private static final String OUTPUT_NODE = "block3_conv1/Relu";

    Encoder(Activity activity) {
        encoderInterface = new TensorFlowInferenceInterface(activity.getAssets(), MODEL_ENCODER_FILE);
    }

    public void run(float featureValues[], float floatValues[], int imgSize) {
        encoderInterface.feed(INPUT_NODE, floatValues,
                1, imgSize, imgSize, 3);
        encoderInterface.run(new String[] {OUTPUT_NODE}, false);
        encoderInterface.fetch(OUTPUT_NODE, featureValues);
    }
}