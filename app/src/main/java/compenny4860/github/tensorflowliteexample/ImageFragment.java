package compenny4860.github.tensorflowliteexample;

//package com.example.android.tflitecamerademo;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.app.Fragment;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.util.LinkedList;

/**
 * A simple {@link Fragment} subclass.
 */
public class ImageFragment extends Fragment {

    private StyleTransfer styleTransfer;
    private static final String TAG = "TfLiteImageClassifier";
    private ImageView styleImageView ;

    /////////////////////////////////////////////////////////////////////////////////
    private final LinkedList<String> mFileList = new LinkedList<>();
    private RecyclerView mRecyclerView;
    private ImageListAdapter mAdapter;
    /////////////////////////////////////////////////////////////////////////////////


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

        /////////////////////////////////////////////////////////////////////////////////////////
        mFileList.addLast("dog.jpg");
        for (int i = 1; i < 27; i++) {
            mFileList.addLast("style" + i + ".jpg");
        }
        // 1. recycler view.
        mRecyclerView = v.findViewById(R.id.recyclerview);
        // 2. adapter
        mAdapter = new ImageListAdapter(getActivity(), mFileList);
        // 3. Link (view -> adaptor)
        mRecyclerView.setAdapter(mAdapter);
        // 4. item들이 표시되는 layout 설정
        LinearLayoutManager horizontalLayoutManagaer = new LinearLayoutManager(getActivity(),
                LinearLayoutManager.HORIZONTAL, false);
        mRecyclerView.setLayoutManager(horizontalLayoutManagaer);
        /////////////////////////////////////////////////////////////////////////////////////////
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
