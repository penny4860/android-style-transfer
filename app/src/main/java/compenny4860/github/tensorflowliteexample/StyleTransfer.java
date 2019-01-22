package compenny4860.github.tensorflowliteexample;

import android.app.Activity;
import android.graphics.Bitmap;
import android.util.Log;
import java.io.IOException;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class StyleTransfer {

    private TensorFlowInferenceInterface inferenceInterface;
    private static final String MODEL_FILE = "file:///android_asset/stylize_quantized.pb";
    private static final String INPUT_NODE = "input";
    private static final String STYLE_NODE = "style_num";
    private static final String OUTPUT_NODE = "transformer/expand/conv3/conv/Sigmoid";
    public static final int NUM_STYLES = 26;

    //    private static final int desiredSize = 256;
//
//    private final float[] styleVals = new float[NUM_STYLES];
    private int[] intValues;
    private float[] floatValues;

    private static final String TAG = "StyleTransferDemo";

    StyleTransfer(Activity activity) throws IOException {
        inferenceInterface = new TensorFlowInferenceInterface(activity.getAssets(), MODEL_FILE);
        setSize(256);

        Log.d(TAG, "Created a TensorFlowInferenceInterface");
    }

    public void setSize(int desiredSize) {
        floatValues = new float[desiredSize * desiredSize * 3];
        intValues = new int[desiredSize * desiredSize];
    }

    public void run(final Bitmap bitmap, float[] styleVals) {
        Log.d(TAG, "style running");
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
        }

        // Copy the input data into TensorFlow.
        inferenceInterface.feed(INPUT_NODE, floatValues,
                1, bitmap.getWidth(), bitmap.getHeight(), 3);
        inferenceInterface.feed(STYLE_NODE, styleVals, NUM_STYLES);

        // Execute the output node's dependency sub-graph.
        inferenceInterface.run(new String[] {OUTPUT_NODE}, false);

        // Copy the data from TensorFlow back into our array.
        inferenceInterface.fetch(OUTPUT_NODE, floatValues);


        for (int i = 0; i < intValues.length; ++i) {
            intValues[i] =
                    0xFF000000
                            | (((int) (floatValues[i * 3] * 255)) << 16)
                            | (((int) (floatValues[i * 3 + 1] * 255)) << 8)
                            | ((int) (floatValues[i * 3 + 2] * 255));
        }
        bitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    }

}

