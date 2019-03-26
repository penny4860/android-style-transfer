package compenny4860.github.tensorflowliteexample;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import java.io.IOException;

public class StyleTransfer {

    private static final String TAG = "StyleTransferDemo";

    private int mImgSize=320;
    private int[] intValues = new int[mImgSize * mImgSize];
    private float[] floatValues = new float[mImgSize * mImgSize * 3];

    private Encoder encoder;
    private Decoder decoder;
    private Mat mat;

    StyleTransfer(Activity activity) throws IOException {
        Log.d(TAG, "Constructor");
        encoder = new Encoder(activity);
        decoder = new Decoder(activity);
        mat = new Mat(activity);
        Log.d(TAG, "Tensorflow model initialized");
    }

    public void setSize(int desiredSize) {
        if (desiredSize != mImgSize) {
            mImgSize = desiredSize;
            intValues = new int[mImgSize * mImgSize];
            floatValues = new float[mImgSize * mImgSize * 3];
        }
    }

    public void run(final Bitmap contentBitmap, final Bitmap styleBitmap) {
        Log.d(TAG, "style running");

        float[] contentFeatureValues = new float[contentBitmap.getWidth()/4 * contentBitmap.getHeight()/4 * 256];
        float[] styleFeatureValues = new float[contentBitmap.getWidth()/4 * contentBitmap.getHeight()/4 * 256];
        float[] csFeatureValues = new float[contentBitmap.getWidth()/4 * contentBitmap.getHeight()/4 * 256];
        float[] stylized_img = new float[contentBitmap.getWidth() * contentBitmap.getHeight() * 3];
        long startTime, endTime;

        // 1. Get contentFeatureValues
        startTime = SystemClock.uptimeMillis();
        getFloatValues(contentBitmap);
        encoder.run(contentFeatureValues, floatValues, mImgSize);

        // 2. Get styleFeatureValues
        getFloatValues(styleBitmap);
        encoder.run(styleFeatureValues, floatValues, mImgSize);
        endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "    1. Timecost to extract features: " + Long.toString(endTime - startTime));

        startTime = SystemClock.uptimeMillis();
        mat.run(csFeatureValues, contentFeatureValues, styleFeatureValues, mImgSize/4);
        endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "    2. Timecost to mixing: " + Long.toString(endTime - startTime));

        // 3. Recon stylized image
        startTime = SystemClock.uptimeMillis();
        decoder.run(stylized_img, csFeatureValues, mImgSize/4);
        endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "    3. Timecost to decoding: " + Long.toString(endTime - startTime));

        // 4. Set bitmap
        setBitmap(contentBitmap, stylized_img);
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

    private void setBitmap(Bitmap contentBitmap, float stylized_img[]) {
        for (int i = 0; i < intValues.length; ++i) {
            intValues[i] =
                    0xFF000000
                            | (((int) (stylized_img[i * 3])) << 16)
                            | (((int) (stylized_img[i * 3 + 1])) << 8)
                            | ((int) (stylized_img[i * 3 + 2]));
        }
        contentBitmap.setPixels(intValues, 0, contentBitmap.getWidth(), 0, 0, contentBitmap.getWidth(), contentBitmap.getHeight());
    }
}

