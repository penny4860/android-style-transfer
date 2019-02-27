package compenny4860.github.tensorflowliteexample;

//package com.example.android.tflitecamerademo;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.app.Fragment;
import android.os.Environment;
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

import static android.app.Activity.RESULT_OK;

/**
 * A simple {@link Fragment} subclass.
 */
public class ImageFragment extends Fragment {

    private StyleTransfer styleTransfer;
    private static final String TAG = "ImageFragment";
    private ImageView styleImageView ;
    public Bitmap contentBitmap;

    /////////////////////////////////////////////////////////////////////////////////
    private final LinkedList<String> mFileList = new LinkedList<>();
    private RecyclerView mRecyclerView;
    private ImageListAdapter mAdapter;
    private TakingPicture mTakingPicture;
    private PickPicture mPickPicture;
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

        // Todo: bitmap 객체를 효율적으로 갖고있는 방법 조사.
        contentBitmap = ((BitmapDrawable)styleImageView.getDrawable()).getBitmap();

        /////////////////////////////////////////////////////////////////////////////////////////
        for (int i = 0; i < 26; i++) {
            mFileList.addLast("style" + i + ".jpg");
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

        mTakingPicture = new TakingPicture();
        mPickPicture = new PickPicture();

        v.findViewById(R.id.take).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent takePictureIntent = mTakingPicture.getTakePhotoIntent(getActivity().getApplicationContext(),
                        getActivity().getPackageManager(), getActivity().getPackageName(),
                        getActivity().getExternalFilesDir(Environment.DIRECTORY_PICTURES));
                startActivityForResult(takePictureIntent, mTakingPicture.getRequestCode());
                Log.d(TAG, "camera button clicked");
            }
        });

        v.findViewById(R.id.pick).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent pickPictureIntent = mPickPicture.getPickIntent();
                startActivityForResult(pickPictureIntent, mPickPicture.getRequestCode());
                Log.d(TAG, "pick button clicked");
            }
        });

        return v;
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {

        Log.d(TAG, "onActivityResult");

        Bitmap bitmap = null;
        if (resultCode == RESULT_OK)
        {
            if (requestCode == mTakingPicture.getRequestCode())
            {
                bitmap = mTakingPicture.getImage();
            }
            else if (requestCode == mPickPicture.getRequestCode())
            {
                Uri imageUri = data.getData();
                styleImageView.setImageURI(imageUri);
                bitmap = ((BitmapDrawable) styleImageView.getDrawable()).getBitmap();
            }
        }

        if (bitmap != null)
        {
            styleImageView.setImageBitmap(bitmap);
            contentBitmap = bitmap;
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
    public void runTransfer(Bitmap contentBitmap, Bitmap styleBitmap) {

        // Bitmap bitmap = ((BitmapDrawable) (imageView.getDrawable())).getBitmap();
        int original_w = contentBitmap.getWidth();
        int original_h = contentBitmap.getHeight();

        Log.d(TAG, "Running." + original_h + ", " + original_w);

        {
            int size = 416;
            styleTransfer.setSize(size);

            contentBitmap = Bitmap.createScaledBitmap(contentBitmap, size, size, true);
            styleBitmap = Bitmap.createScaledBitmap(styleBitmap, size, size, true);

            styleTransfer.run(contentBitmap, styleBitmap);
            contentBitmap = Bitmap.createScaledBitmap(contentBitmap, original_w, original_h, true);
        }
        styleImageView.setImageBitmap(contentBitmap);
    }


}
