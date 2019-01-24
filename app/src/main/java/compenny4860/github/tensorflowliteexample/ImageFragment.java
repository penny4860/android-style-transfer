package compenny4860.github.tensorflowliteexample;

//package com.example.android.tflitecamerademo;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.app.Fragment;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;

/**
 * A simple {@link Fragment} subclass.
 */
public class ImageFragment extends Fragment {

    private StyleTransfer styleTransfer;
    private static final String TAG = "TfLiteImageClassifier";
    private ImageView imageView ;
    private ImageView styleImageView ;
    private ImageView styleView0;
    private ImageView styleView1;
    private ImageView styleView2;

    public ImageFragment() {
        // Required empty public constructor
    }

    public static ImageFragment newInstance() {
        return new ImageFragment();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment

        View v = inflater.inflate(R.layout.fragment_image, container, false);
        styleImageView = v.findViewById(R.id.styleImageView);

        imageView = v.findViewById(R.id.imageView);
        styleView0 = v.findViewById(R.id.styleView0);
        styleView1 = v.findViewById(R.id.styleView1);
        styleView2 = v.findViewById(R.id.styleView2);

        imageView.setOnClickListener(new StyleListener(-1));
        styleView0.setOnClickListener(new StyleListener(0));
        styleView1.setOnClickListener(new StyleListener(1));
        styleView2.setOnClickListener(new StyleListener(2));

        return v;
    }

    class StyleListener implements View.OnClickListener {

        private int mStyleIndex;

        public StyleListener(int styleIndex)
        {
            mStyleIndex = styleIndex;
        }

        @Override
        public void onClick(View v) {
            runTransfer(mStyleIndex);
        }
    }

    /** Load the model and labels. */
    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        try {
            styleTransfer = new StyleTransfer(getActivity());

        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize an image classifier.");
        }
    }

    /** Classifies a frame from the preview stream. */
    private void runTransfer(int styleIndex) {

        Bitmap bitmap = ((BitmapDrawable) (imageView.getDrawable())).getBitmap();
        int original_w = bitmap.getWidth();
        int original_h = bitmap.getHeight();

        Log.d(TAG, "Running." + original_h + ", " + original_w);

        if (styleIndex == -1)
        {

        }
        else
        {
            int size = 256;
            styleTransfer.setSize(size);

            bitmap = Bitmap.createScaledBitmap(bitmap, size, size, true);
            float[] styleValues = new float[styleTransfer.NUM_STYLES];
            styleValues[styleIndex] = 1.0f;
            styleTransfer.run(bitmap, styleValues);
            bitmap = Bitmap.createScaledBitmap(bitmap, original_w, original_h, true);
        }
        styleImageView.setImageBitmap(bitmap);
    }


}
