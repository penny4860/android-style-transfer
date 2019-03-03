package compenny4860.github.tensorflowliteexample;

import android.app.ProgressDialog;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.os.AsyncTask;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;


public class ImageListAdapter extends RecyclerView.Adapter<ImageListAdapter.ImageViewHolder> {

    private LinkedList<String> mFileList;
    private LayoutInflater mInflater;
    private Context mContext;
    private ImageFragment mImageFragment;

    // 1. Adapter의 constructor 구현
    //      1) layout inflator : resource에서 정의한 view를 가져와서 element view를 설정
    //      2) dataset : ListView에 표시할 data 설정
    public ImageListAdapter(Context context, LinkedList<String> fileList, ImageFragment imageFragment)
    {
        mInflater = LayoutInflater.from(context);
        mFileList = fileList;
        mContext = context;
        mImageFragment = imageFragment;
    }

    // 2. Element View(ViewHolder) 가 생성될때 실행되는 method
    @Override
    public ImageListAdapter.ImageViewHolder onCreateViewHolder(ViewGroup viewGroup, int viewType) {
        // view 객체를 생성.
        View mItemView = mInflater.inflate(R.layout.style_element, viewGroup, false);
        return new ImageViewHolder(mItemView);
    }

    // 3. ViewHolder 가 서로 바인딩될 때 : index를 얻어서 뷰에 데이터를 설정
    @Override
    public void onBindViewHolder(ImageListAdapter.ImageViewHolder imageViewHolder, int posIndex) {
        String imgFileName = mFileList.get(posIndex);
        String styleName = imgFileName.substring(0, imgFileName.lastIndexOf('.'));

        imageViewHolder.imageTextView.setText(styleName);
        Bitmap bitmap = null;
        try {
            bitmap = getImage(imgFileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
        imageViewHolder.imageView.setImageBitmap(bitmap);
    }

    @Override
    public int getItemCount() {
        return mFileList.size();
    }

    // ViewHolder class 정의
    class ImageViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {
        TextView imageTextView;
        ImageView imageView;
        public ImageViewHolder(View itemView) {
            super(itemView);
            imageTextView = itemView.findViewById(R.id.imageFilename);
            imageView = itemView.findViewById(R.id.imageView);
            itemView.setOnClickListener(this);
        }

        @Override
        public void onClick(View view) {
            // Get the position of the item that was clicked.
            int mPosition = getLayoutPosition();
            // Toast.makeText(mContext, mPosition + " clicked", Toast.LENGTH_LONG).show();

            Bitmap contentBitmap = null;
            Bitmap styleBitmap = null;
            try {
                contentBitmap = mImageFragment.contentBitmap;
                styleBitmap = getImage(mFileList.get(mPosition));

            } catch (IOException e) {
                e.printStackTrace();
            }

            CheckTypesTask task = new CheckTypesTask(contentBitmap, styleBitmap);
            task.execute();
        }
    }

    private Bitmap getImage(String filename) throws IOException {
        AssetManager assetManager = mContext.getAssets();
        InputStream istr = null;
        try {
            istr = assetManager.open(filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Bitmap bitmap = BitmapFactory.decodeStream(istr);
        return bitmap;
    }

    private class CheckTypesTask extends AsyncTask<Void, Void, Void> {

        ProgressDialog asyncDialog = new ProgressDialog(
                mContext);

        Bitmap contentBitmap;
        Bitmap styleBitmap;

        CheckTypesTask(Bitmap c, Bitmap s)
        {
            contentBitmap = c;
            styleBitmap = s;
        }


        @Override
        protected void onPreExecute() {
            asyncDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
            asyncDialog.setMessage("Filtering..");

            // show dialog
            asyncDialog.show();
            super.onPreExecute();
        }

        @Override
        protected Void doInBackground(Void... arg0) {
            mImageFragment.runTransfer(contentBitmap, styleBitmap);
            return null;
        }

        @Override
        protected void onPostExecute(Void result) {
            asyncDialog.dismiss();
            super.onPostExecute(result);
        }
    }
}

