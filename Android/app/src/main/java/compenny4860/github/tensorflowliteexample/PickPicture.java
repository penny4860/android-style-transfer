package compenny4860.github.tensorflowliteexample;

import android.content.ContentResolver;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.provider.MediaStore;

import java.io.IOException;

public class PickPicture {


    /////////////////////////////////////////////////////////////////////////////////////////////////
    private static final int PICK_IMAGE = 100;
    public Intent getPickIntent() {
        Intent gallery =
                new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        return gallery;
    }

    public Bitmap getImage(Intent data, ContentResolver contentResolver) {
        Uri imageUri = data.getData();
        Bitmap bitmap = null;
        try {
            bitmap = MediaStore.Images.Media.getBitmap(contentResolver, imageUri);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bitmap;
    }

    public int getRequestCode() {
        return PICK_IMAGE;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////


}
