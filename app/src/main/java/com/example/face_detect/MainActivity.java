package com.example.face_detect;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.text.Html;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;


import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {

    private static final String TAG = "OpenCameraActivity";
    private Mat mRgba;
    private Mat mYcrcb;
    private Mat mGray;
    private Mat result_img;
    private Mat cr1;
    private Mat zero_mat;
    private Mat mask_roi;
//    private CameraBridgeViewBase mOpenCvCameraView;
    private JavaCameraView mOpenCvCameraView;
    private static final int MY_CAMERA_REQUEST_CODE = 100;

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    static {
        OpenCVLoader.initDebug();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.camera_surface_view);
        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.fd_activity_surface_view);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Permissions granted");
            initializeCamera();
        } else {
            Log.d(TAG, "Troubles");
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
        }

    }
    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera Permission granted", Toast.LENGTH_LONG).show();
                initializeCamera();
            } else {
                Toast.makeText(this, "Camera Permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }

    private void initializeCamera(){
        mOpenCvCameraView.enableView();
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);//前置攝像頭 CameraBridgeViewBase.CAMERA_ID_BACK爲後置攝像頭
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mYcrcb = new Mat();
        result_img = new Mat();
        mGray = new Mat();
        cr1 = new Mat();
        zero_mat = Mat.zeros(height, width, CvType.CV_8U);
        mask_roi = Mat.zeros(height, width, CvType.CV_8U);
        Imgproc.rectangle(mask_roi, new Point(width*2/9, height*1/6), new Point(width*7/9, height*5/6), Scalar.all(255), -1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mYcrcb.release();
        result_img.release();
        mGray.release();
        cr1.release();
        zero_mat.release();
        mask_roi.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        double area;
        int width, height;
        mRgba = inputFrame.rgba();
        Size originSize = mRgba.size();

        // Transpose Image and flip Image
        Core.transpose(mRgba, mRgba);
        Core.flip(mRgba, mRgba, -1);
        Imgproc.resize(mRgba, mRgba, originSize);
        width = mRgba.cols();
        height = mRgba.rows();

        // convert to Ycrcb from rgba
        Imgproc.cvtColor(mRgba, mYcrcb, Imgproc.COLOR_RGB2YCrCb);
        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGB2GRAY);
        //split 3 channels
        List<Mat> ycrcb_list = new ArrayList(3);
        Core.split(mYcrcb, ycrcb_list);

        // gaussian blur for cr channel
        Imgproc.GaussianBlur(ycrcb_list.get(1), cr1, new Size(5,5), 0);
        Imgproc.threshold(cr1, cr1, 0, 1, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
        // Imgproc.erode(cr1, cr1, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5)), new Point(-1, -1), 5);
        // Imgproc.dilate(cr1, cr1, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5)), new Point(-1, -1),2);

        // bitwise operation
//        Core.inRange(mGray, new Scalar(200), new Scalar(255), mGray);
//        Core.bitwise_not(mGray, mGray);
        Core.bitwise_and(cr1, mask_roi, cr1);
//        Core.bitwise_and(cr1, mGray, cr1);

        area = Math.round(Core.sumElems(cr1).val[0] / (width * height) * 1000.0) / 1000.0;
        Core.multiply(cr1, new Scalar(255), cr1);
        ycrcb_list.set(1, cr1);
        ycrcb_list.set(0, zero_mat);
        ycrcb_list.set(2, zero_mat);
        Core.merge(ycrcb_list, result_img);

        Imgproc.putText(result_img, "Area: " + String.valueOf(area), new Point(10, 100), Core.FONT_HERSHEY_COMPLEX, 1.0, new Scalar(255,0,0), 2);
        Imgproc.putText(result_img, "Posture: ", new Point(10, 180), Core.FONT_HERSHEY_COMPLEX, 1.0, new Scalar(255,0,0), 2);

        mRgba.release();
        mYcrcb.release();
        cr1.release();
        ycrcb_list.clear();
        mGray.release();
        return result_img;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        return true;
    }
}