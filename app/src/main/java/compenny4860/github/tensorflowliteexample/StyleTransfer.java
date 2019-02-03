package compenny4860.github.tensorflowliteexample;

import android.app.Activity;
import android.graphics.Bitmap;
import android.util.Log;
import java.io.IOException;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class StyleTransfer {

    private TensorFlowInferenceInterface encoderInterface;
    private TensorFlowInferenceInterface decoderInterface;

    private static final String MODEL_FILE = "encoder_opt.pb";

    private static final String INPUT_NODE = "input";
    private static final String STYLE_NODE = "style_num";
    private static final String OUTPUT_NODE = "output/Relu";
    public static final int NUM_STYLES = 26;

    //    private static final int desiredSize = 256;
//
//    private final float[] styleVals = new float[NUM_STYLES];
    private int[] intValues;
    private float[] floatValues;

    private static final String TAG = "StyleTransferDemo";

    StyleTransfer(Activity activity) throws IOException {
        Log.d(TAG, "Constructor");
        encoderInterface = new TensorFlowInferenceInterface(activity.getAssets(), MODEL_FILE);
        Log.d(TAG, "Created a TensorFlowInferenceInterface 1");
        decoderInterface = new TensorFlowInferenceInterface(activity.getAssets(), "decoder_opt.pb");
        Log.d(TAG, "Created a TensorFlowInferenceInterface 2");
        setSize(256);

    }

    public void setSize(int desiredSize) {
        //Todo: desiredSize가 기존 value와 다를 때만 array를 새로 생성
        floatValues = new float[desiredSize * desiredSize * 3];
        intValues = new int[desiredSize * desiredSize];
    }

    private void getFloatValues(final Bitmap bitmap) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
        }
    }

    private void getFeatures(final Bitmap bitmap, float featureValues[]) {
        encoderInterface.feed(INPUT_NODE, floatValues,
                1, bitmap.getWidth(), bitmap.getHeight(), 3);
        encoderInterface.run(new String[] {OUTPUT_NODE}, false);
        encoderInterface.fetch(OUTPUT_NODE, featureValues);
    }

    public void run(final Bitmap contentBitmap, final Bitmap styleBitmap) {
        Log.d(TAG, "style running");

        float[] contentFeatureValues = new float[contentBitmap.getWidth()/8 * contentBitmap.getHeight()/8 * 512];
        float[] styleFeatureValues = new float[contentBitmap.getWidth()/8 * contentBitmap.getHeight()/8 * 512];
        float[] stylized_img = new float[contentBitmap.getWidth() * contentBitmap.getHeight() * 3];

        // 1. Get contentFeatureValues
        getFloatValues(contentBitmap);
        getFeatures(contentBitmap, contentFeatureValues);

        // 2. Get styleFeatureValues
        getFloatValues(styleBitmap);
        getFeatures(styleBitmap, styleFeatureValues);
        Log.d(TAG, "encoder running is done");

        decoderInterface.feed("input_c", contentFeatureValues,
                1, contentBitmap.getWidth()/8, contentBitmap.getHeight()/8, 512);
        decoderInterface.feed("input_s", styleFeatureValues,
                1, styleBitmap.getWidth()/8, styleBitmap.getHeight()/8, 512);
        decoderInterface.run(new String[] {"output/mul"}, false);
        decoderInterface.fetch(OUTPUT_NODE, stylized_img);
        Log.d(TAG, "decoder running is done");

        for (int i = 0; i < intValues.length; ++i) {
            intValues[i] =
                    0xFF000000
                            | (((int) (stylized_img[i * 3] * 255)) << 16)
                            | (((int) (stylized_img[i * 3 + 1] * 255)) << 8)
                            | ((int) (stylized_img[i * 3 + 2] * 255));
        }
        contentBitmap.setPixels(intValues, 0, contentBitmap.getWidth(), 0, 0, contentBitmap.getWidth(), contentBitmap.getHeight());
        Log.d(TAG, "set bitmap");

    }

}

