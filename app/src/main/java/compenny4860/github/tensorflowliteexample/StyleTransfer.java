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
        decoderInterface = new TensorFlowInferenceInterface(activity.getAssets(), "decoder_opt.pb");
        setSize(256);
        testPorting();
        Log.d(TAG, "Test Porting is done");
    }

    private void testPorting()
    {
        float[] inputValues = {118.828125f, 173.82812f, 228.79688f, 125.59375f, 177.90625f, 227.5625f, 133.4375f, 184.0625f, 234.75f, 117.3125f, 176.3125f, 232.3125f, 142.39062f, 189.8125f, 234.26562f, 135.75f, 187.75f, 234.75f, 146.48438f, 191.48438f, 233.48438f, 128.92188f, 184.40625f, 233.57812f, 132.40625f, 185.65625f, 234.90625f, 130.3125f, 184.3125f, 231.3125f, 139.51562f, 187.51562f, 233.51562f, 123.171875f, 178.51562f, 232.84375f, 121.78125f, 179.14062f, 232.07812f, 119.765625f, 176.51562f, 233.26562f, 116.03125f, 172.03125f, 229.03125f, 121.484375f, 173.23438f, 229.73438f, 140.10938f, 189.0625f, 232.20312f, 121.1875f, 181.0625f, 230.9375f, 123.703125f, 183.76562f, 233.57812f, 120.734375f, 180.46875f, 235.04688f, 127.0625f, 184.96875f, 235.01562f, 137.89062f, 189.89062f, 236.89062f, 152.35938f, 195.35938f, 237.35938f, 129.04688f, 184.90625f, 235.92188f, 159.125f, 197.1875f, 236.28125f, 161.9375f, 198.9375f, 240.9375f, 188.07812f, 214.07812f, 241.07812f, 138.73438f, 189.76562f, 236.25f, 126.4375f, 183.9375f, 233.1875f, 127.28125f, 183.28125f, 232.28125f, 149.625f, 191.625f, 232.125f, 163.32812f, 198.0f, 235.71875f, 150.23438f, 191.25f, 230.92188f, 127.0f, 184.25f, 229.5f, 125.890625f, 185.89062f, 237.01562f, 138.57812f, 191.85938f, 237.125f, 148.0f, 195.625f, 238.71875f, 138.57812f, 190.57812f, 237.57812f, 143.25f, 194.25f, 237.25f, 157.15625f, 198.96875f, 237.96875f, 163.125f, 202.125f, 241.125f, 196.75f, 219.625f, 243.0625f, 196.28125f, 219.90625f, 241.70312f, 180.28125f, 208.54688f, 239.92188f, 171.76562f, 205.20312f, 239.35938f, 164.4375f, 202.07812f, 240.76562f, 185.65625f, 212.60938f, 240.4375f, 191.375f, 214.25f, 239.125f, 149.71875f, 192.09375f, 230.28125f, 146.79688f, 194.17188f, 235.98438f, 142.53125f, 193.90625f, 238.71875f, 145.95312f, 196.95312f, 240.23438f, 138.375f, 195.65625f, 240.51562f, 144.59375f, 197.59375f, 241.59375f, 143.0625f, 196.20312f, 240.20312f, 141.0625f, 195.0625f, 241.0625f, 141.57812f, 196.1875f, 241.89062f, 186.79688f, 218.17188f, 247.54688f, 222.625f, 238.25f, 250.875f, 228.67188f, 240.92188f, 250.26562f, 192.79688f, 218.89062f, 241.9375f, 189.25f, 215.25f, 240.25f, 179.92188f, 209.32812f, 239.625f, 206.20312f, 227.20312f, 246.57812f, 127.609375f, 186.60938f, 230.60938f, 131.04688f, 192.04688f, 236.04688f, 142.46875f, 200.04688f, 240.1875f, 146.0625f, 200.5625f, 241.8125f, 145.42188f, 199.21875f, 244.71875f, 144.625f, 199.1875f, 245.0f, 153.875f, 202.875f, 243.875f, 142.51562f, 198.76562f, 244.01562f, 148.70312f, 194.54688f, 232.34375f, 146.20312f, 199.20312f, 243.20312f, 198.21875f, 226.07812f, 247.28125f, 208.4375f, 229.4375f, 246.8125f, 204.42188f, 225.8125f, 247.09375f, 170.4375f, 206.4375f, 240.84375f, 214.76562f, 232.71875f, 246.8125f, 236.54688f, 243.54688f, 249.54688f, 136.78125f, 191.78125f, 232.78125f, 135.54688f, 193.54688f, 233.54688f, 143.04688f, 201.04688f, 241.04688f, 142.0f, 201.0f, 243.0f, 143.5f, 200.5f, 243.5f, 144.4375f, 202.5625f, 245.0f, 155.0f, 207.0f, 247.0f, 160.54688f, 209.20312f, 245.75f, 156.82812f, 181.25f, 205.6875f, 223.3125f, 241.20312f, 251.82812f, 248.10938f, 252.10938f, 253.10938f, 223.4375f, 238.3125f, 251.6875f, 211.96875f, 230.01562f, 247.29688f, 238.32812f, 245.32812f, 253.32812f, 242.0f, 247.0f, 251.0f, 241.3125f, 245.23438f, 249.14062f, 148.875f, 198.75f, 235.21875f, 146.34375f, 198.34375f, 235.34375f, 177.67188f, 212.42188f, 240.17188f, 153.375f, 206.875f, 245.125f, 150.07812f, 205.07812f, 246.07812f, 151.04688f, 206.04688f, 247.04688f, 153.29688f, 208.29688f, 249.29688f, 151.5f, 206.5f, 247.0f, 34.3125f, 50.3125f, 65.3125f, 158.21875f, 211.20312f, 245.25f, 221.78125f, 238.82812f, 246.92188f, 215.10938f, 234.0625f, 248.0625f, 236.59375f, 244.0625f, 249.82812f, 252.1875f, 253.1875f, 248.1875f, 239.9375f, 244.9375f, 247.9375f, 239.3125f, 244.07812f, 247.07812f, 198.0625f, 220.75f, 236.125f, 176.82812f, 211.14062f, 235.20312f, 153.76562f, 206.76562f, 238.07812f, 157.40625f, 210.40625f, 243.03125f, 158.17188f, 211.17188f, 245.17188f, 156.85938f, 211.48438f, 247.17188f, 158.25f, 212.0f, 247.5f, 163.67188f, 214.20312f, 247.59375f, 24.453125f, 40.953125f, 54.671875f, 180.59375f, 220.625f, 248.71875f, 220.79688f, 238.60938f, 247.14062f, 239.15625f, 244.90625f, 248.17188f, 240.82812f, 245.82812f, 248.82812f, 236.3125f, 243.375f, 246.375f, 237.20312f, 241.51562f, 243.14062f, 227.23438f, 236.23438f, 243.85938f, 161.07812f, 206.07812f, 235.07812f, 161.0f, 208.0f, 236.625f, 167.3125f, 211.3125f, 238.65625f, 172.98438f, 214.92188f, 243.23438f, 166.48438f, 215.48438f, 246.51562f, 165.48438f, 215.85938f, 245.79688f, 173.5625f, 215.5625f, 240.15625f, 75.390625f, 84.75f, 93.828125f, 33.234375f, 49.234375f, 65.15625f, 195.40625f, 227.3125f, 246.0f, 235.0625f, 243.375f, 247.0f, 38.84375f, 48.78125f, 64.96875f, 239.0f, 245.0f, 245.0f, 236.6875f, 241.6875f, 244.6875f, 159.65625f, 76.140625f, 75.921875f, 36.109375f, 25.484375f, 36.171875f, 167.29688f, 212.29688f, 235.29688f, 168.21875f, 213.21875f, 236.21875f, 170.0f, 215.0f, 234.0f, 173.54688f, 217.79688f, 240.54688f, 174.375f, 222.375f, 242.375f, 183.40625f, 223.79688f, 243.48438f, 61.1875f, 84.734375f, 108.359375f, 85.015625f, 93.53125f, 100.234375f, 30.453125f, 38.359375f, 49.875f, 229.375f, 240.75f, 246.79688f, 128.9375f, 160.70312f, 180.82812f, 41.96875f, 49.21875f, 64.46875f, 78.90625f, 86.6875f, 96.953125f, 225.0f, 234.0f, 239.0f, 120.203125f, 45.328125f, 46.078125f, 26.8125f, 20.859375f, 30.875f, 180.1875f, 215.85938f, 229.96875f, 189.95312f, 218.95312f, 226.95312f, 192.39062f, 221.54688f, 231.0f, 127.21875f, 114.21875f, 103.46875f, 192.8125f, 225.6875f, 238.25f, 90.78125f, 75.796875f, 74.9375f, 63.921875f, 84.5f, 103.640625f, 70.59375f, 82.84375f, 91.234375f, 178.34375f, 165.04688f, 150.35938f, 226.3125f, 238.0625f, 240.3125f, 67.53125f, 80.609375f, 95.625f, 151.34375f, 145.25f, 139.28125f, 59.84375f, 69.125f, 75.703125f, 44.21875f, 51.0f, 61.765625f, 148.67188f, 63.03125f, 60.71875f, 36.0625f, 23.0625f, 32.8125f, 43.375f, 39.09375f, 40.234375f, 71.765625f, 66.0625f, 63.671875f, 119.34375f, 102.84375f, 93.96875f, 14.859375f, 17.90625f, 26.90625f, 61.84375f, 66.375f, 67.875f, 66.203125f, 75.296875f, 80.25f, 62.421875f, 73.921875f, 85.921875f, 80.578125f, 87.546875f, 87.765625f, 168.23438f, 158.6875f, 145.03125f, 34.984375f, 44.984375f, 54.984375f, 8.84375f, 20.84375f, 32.84375f, 142.5f, 135.9375f, 127.84375f, 43.0625f, 50.5625f, 58.8125f, 154.0f, 146.17188f, 134.6875f, 91.296875f, 45.125f, 44.4375f, 59.375f, 58.625f, 56.4375f, 19.890625f, 19.1875f, 25.546875f, 43.65625f, 41.09375f, 42.65625f, 93.21875f, 69.40625f, 60.84375f, 6.59375f, 11.15625f, 21.59375f, 22.859375f, 21.859375f, 29.859375f, 21.0625f, 24.609375f, 29.125f, 52.359375f, 51.03125f, 56.6875f, 38.390625f, 50.921875f, 58.890625f, 127.046875f, 108.65625f, 94.921875f, 39.28125f, 43.34375f, 46.96875f, 56.21875f, 60.46875f, 64.46875f, 98.28125f, 84.703125f, 75.6875f, 126.171875f, 122.375f, 112.15625f, 125.609375f, 110.546875f, 98.875f, 53.328125f, 32.71875f, 34.796875f, 113.265625f, 96.15625f, 83.4375f, 77.1875f, 97.96875f, 103.078125f, 76.65625f, 76.90625f, 75.65625f, 80.1875f, 72.109375f, 57.484375f, 5.140625f, 12.140625f, 19.078125f, 21.296875f, 23.796875f, 23.796875f, 76.09375f, 72.796875f, 49.6875f, 48.578125f, 49.828125f, 56.390625f, 31.703125f, 37.390625f, 45.828125f, 125.796875f, 104.625f, 79.734375f, 219.90625f, 196.34375f, 167.03125f, 67.3125f, 70.140625f, 62.375f, 93.0625f, 83.796875f, 64.34375f, 52.515625f, 50.875f, 41.515625f, 87.984375f, 84.984375f, 69.984375f, 48.75f, 51.140625f, 42.65625f, 103.625f, 91.046875f, 82.328125f, 67.953125f, 86.546875f, 96.34375f, 28.09375f, 30.0f, 16.34375f, 63.1875f, 61.1875f, 46.1875f, 66.421875f, 63.453125f, 46.3125f, 53.8125f, 52.390625f, 47.109375f, 65.25f, 68.546875f, 47.5f, 48.625f, 49.765625f, 32.953125f, 48.8125f, 49.8125f, 34.71875f, 73.53125f, 70.890625f, 51.453125f, 33.65625f, 35.78125f, 21.953125f, 27.0625f, 29.1875f, 16.125f, 28.796875f, 26.5f, 20.640625f, 70.71875f, 59.09375f, 45.640625f, 42.53125f, 39.59375f, 23.1875f, 66.28125f, 61.1875f, 41.28125f, 70.8125f, 65.0f, 49.34375f, 77.890625f, 76.109375f, 68.0f, 164.23438f, 156.17188f, 141.48438f, 125.796875f, 111.171875f, 98.578125f, 108.28125f, 146.85938f, 152.35938f, 102.25f, 150.21875f, 159.46875f, 102.53125f, 153.03125f, 159.75f, 99.609375f, 151.9375f, 159.29688f, 97.53125f, 146.17188f, 149.90625f, 92.125f, 145.25f, 150.35938f, 99.59375f, 150.03125f, 159.46875f, 100.28125f, 152.20312f, 160.23438f, 97.359375f, 149.60938f, 155.04688f, 94.71875f, 148.5f, 155.5f, 99.140625f, 148.20312f, 157.92188f, 95.078125f, 141.51562f, 150.73438f, 92.328125f, 140.45312f, 150.82812f, };
        float[] featValues = new float[2*2*512];

        Log.d(TAG, inputValues.length + ", ");

        encoderInterface.feed(INPUT_NODE, inputValues,
                1, 16, 16, 3);
        encoderInterface.run(new String[] {OUTPUT_NODE}, false);
        encoderInterface.fetch(OUTPUT_NODE, featValues);

        String msg = "";
        for (int i=0; i < featValues.length; i++)
        {
            msg = msg + featValues[i] + " ,";
        }
        Log.d(TAG, msg);

        decoderInterface.feed("input_c", featValues,
                1, 2, 2, 512);
        decoderInterface.feed("input_s", featValues,
                1, 2, 2, 512);
        decoderInterface.run(new String[] {"output/mul"}, false);
        decoderInterface.fetch("output/mul", inputValues);

        msg = "";
        for (int i=0; i < inputValues.length; i++)
        {
            msg = msg + inputValues[i] + " ,";
        }
        Log.d(TAG, msg);
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

        Log.d(TAG, "content size: " + contentBitmap.getWidth() + ", " + contentBitmap.getHeight() + ", " + contentFeatureValues.length);
        Log.d(TAG, "style size: " + styleBitmap.getWidth() + ", " + styleBitmap.getHeight() + ", " + styleFeatureValues.length);

        decoderInterface.feed("input_c", contentFeatureValues,
                1, contentBitmap.getWidth()/8, contentBitmap.getHeight()/8, 512);
        decoderInterface.feed("input_s", styleFeatureValues,
                1, styleBitmap.getWidth()/8, styleBitmap.getHeight()/8, 512);
        decoderInterface.run(new String[] {"output/mul"}, false);
        Log.d(TAG, "style size: " + stylized_img.length);
        Log.d(TAG, "decoder running.......");
        decoderInterface.fetch("output/mul", stylized_img);
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

