package compenny4860.github.tensorflowliteexample;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import java.io.IOException;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class StyleTransfer {

    public class Encoder {
        private TensorFlowInferenceInterface encoderInterface;
        private static final String MODEL_ENCODER_FILE = "mobile_encoder_opt.pb";
        private static final String INPUT_NODE = "input";
        private static final String OUTPUT_NODE = "output/Relu";

        Encoder(Activity activity) {
             encoderInterface = new TensorFlowInferenceInterface(activity.getAssets(), MODEL_ENCODER_FILE);
        }

        public void run(float featureValues[], int imgSize) {
            encoderInterface.feed(INPUT_NODE, floatValues,
                    1, imgSize, imgSize, 3);
            encoderInterface.run(new String[] {OUTPUT_NODE}, false);
            encoderInterface.fetch(OUTPUT_NODE, featureValues);
        }
    }

    private TensorFlowInferenceInterface decoderInterface;
    private static final String MODEL_DECODER_FILE = "decoder_opt.pb";
    private static final int DEFAULT_SIZE = 256;

    private int[] intValues;
    private float[] floatValues;
    private static final String TAG = "StyleTransferDemo";

    private Encoder encoder;

    StyleTransfer(Activity activity) throws IOException {
        Log.d(TAG, "Constructor");

        encoder = new Encoder(activity);
        decoderInterface = new TensorFlowInferenceInterface(activity.getAssets(), MODEL_DECODER_FILE);
        setSize(DEFAULT_SIZE);
        Log.d(TAG, "Tensorflow model initialized");
    }

    public void setSize(int desiredSize) {
        //Todo: desiredSize가 기존 value와 다를 때만 array를 새로 생성
        floatValues = new float[desiredSize * desiredSize * 3];
        intValues = new int[desiredSize * desiredSize];
    }

    public void run(final Bitmap contentBitmap, final Bitmap styleBitmap) {
        Log.d(TAG, "style running");

        float[] contentFeatureValues = new float[contentBitmap.getWidth()/8 * contentBitmap.getHeight()/8 * 512];
        float[] styleFeatureValues = new float[contentBitmap.getWidth()/8 * contentBitmap.getHeight()/8 * 512];
        float[] stylized_img = new float[contentBitmap.getWidth() * contentBitmap.getHeight() * 3];
        long startTime, endTime;

        // 1. Get contentFeatureValues
        startTime = SystemClock.uptimeMillis();
        getFloatValues(contentBitmap);
        getFeatures(contentBitmap, contentFeatureValues);

        // 2. Get styleFeatureValues
        getFloatValues(styleBitmap);
        getFeatures(styleBitmap, styleFeatureValues);
        endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "    1. Timecost to extract features: " + Long.toString(endTime - startTime));

        startTime = SystemClock.uptimeMillis();
        decoderInterface.feed("input_c", contentFeatureValues,
                1, contentBitmap.getWidth()/8, contentBitmap.getHeight()/8, 512);
        decoderInterface.feed("input_s", styleFeatureValues,
                1, styleBitmap.getWidth()/8, styleBitmap.getHeight()/8, 512);
        decoderInterface.run(new String[] {"output/mul"}, false);
        decoderInterface.fetch("output/mul", stylized_img);
        endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "    2. Timecost to decoding: " + Long.toString(endTime - startTime));

        for (int i = 0; i < intValues.length; ++i) {
            intValues[i] =
                    0xFF000000
                            | (((int) (stylized_img[i * 3])) << 16)
                            | (((int) (stylized_img[i * 3 + 1])) << 8)
                            | ((int) (stylized_img[i * 3 + 2]));
        }
        contentBitmap.setPixels(intValues, 0, contentBitmap.getWidth(), 0, 0, contentBitmap.getWidth(), contentBitmap.getHeight());
        Log.d(TAG, "set bitmap");

    }

    private void getFloatValues(final Bitmap bitmap) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) / 1.0f;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 1.0f;
            floatValues[i * 3 + 2] = (val & 0xFF) / 1.0f;
        }
    }

    private void getFeatures(final Bitmap bitmap, float featureValues[]) {
        encoder.run(featureValues, bitmap.getWidth());
    }
}

