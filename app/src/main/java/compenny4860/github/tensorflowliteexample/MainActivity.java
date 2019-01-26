package compenny4860.github.tensorflowliteexample;

import android.app.Activity;
import android.os.Bundle;

/** Main {@code Activity} class for the Camera app. */
public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        if (null == savedInstanceState) {

            // 런타임에 액티비티에 프래그먼트 추가하는 과정 : 특정 Resource에 프래그먼트를 추가
            // R.id.container : 리소스파일에서 정의
            getFragmentManager()
                    .beginTransaction()
                    .replace(R.id.container, ImageFragment.newInstance())
                    .commit();

        }
    }
}
