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
            mFileList.addLast("style" + (i-1) + ".jpg");
        }
        // 1. recycler view.
        mRecyclerView = v.findViewById(R.id.recyclerview);
        // 2. adapter
        // Todo : 본 객체를 넘기지 말고 옵저버패턴으로 바꿔보자.
        mAdapter = new ImageListAdapter(getActivity(), mFileList, this);
        // 3. Link (view -> adaptor)
        mRecyclerView.setAdapter(mAdapter);
        // 4. item들이 표시되는 layout 설정
        LinearLayoutManager horizontalLayoutManagaer = new LinearLayoutManager(getActivity(),
                LinearLayoutManager.HORIZONTAL, false);
        mRecyclerView.setLayoutManager(horizontalLayoutManagaer);
        /////////////////////////////////////////////////////////////////////////////////////////
        return v;
    }

    /** Load the model and labels. */
    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        try {
            Bitmap bitmap = ((BitmapDrawable) (styleImageView.getDrawable())).getBitmap();
            bitmap = Bitmap.createScaledBitmap(bitmap, 32, 32, true);
            styleTransfer = new StyleTransfer(getActivity(), bitmap);
            styleImageView.setImageBitmap(bitmap);

        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }

    /** Classifies a frame from the preview stream. */
    public void runTransfer(Bitmap contentBitmap, Bitmap styleBitmap) {

        // Bitmap bitmap = ((BitmapDrawable) (imageView.getDrawable())).getBitmap();
        int original_w = contentBitmap.getWidth();
        int original_h = contentBitmap.getHeight();

        Log.d(TAG, "Running." + original_h + ", " + original_w);

        {
            int size = 128;
            styleTransfer.setSize(size);

            contentBitmap = Bitmap.createScaledBitmap(contentBitmap, size, size, true);
            styleBitmap = Bitmap.createScaledBitmap(styleBitmap, size, size, true);

            styleTransfer.run(contentBitmap, styleBitmap);
            contentBitmap = Bitmap.createScaledBitmap(contentBitmap, original_w, original_h, true);
        }
        styleImageView.setImageBitmap(contentBitmap);
    }


}
