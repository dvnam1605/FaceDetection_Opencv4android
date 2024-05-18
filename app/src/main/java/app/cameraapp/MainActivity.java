package app.cameraapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.TorchState;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "FaceDetection";
    ImageButton capture, toggleFlash, switchCamera;
    private PreviewView previewView;
    private FaceDetectorYN faceDetector;
    int lensFacing = CameraSelector.LENS_FACING_BACK;
    Camera camera;
    float mScale = 1.0f;
    private Size mInputSize = null;
    private final ActivityResultLauncher<String> activityResultLauncher = registerForActivityResult(new ActivityResultContracts.RequestPermission(), o -> {
        if(ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
            startCamera();
        }
    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        loadOpenCV();
        loadYunetModel();

        previewView = findViewById(R.id.previewView);
        capture = findViewById(R.id.recordButton);
        toggleFlash = findViewById(R.id.flashButton);
        switchCamera = findViewById(R.id.switchCameraButton);


        if(ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
            startCamera();
        }else{
            activityResultLauncher.launch(Manifest.permission.CAMERA);
        }

        switchCamera.setOnClickListener(v -> switchCamera());
        toggleFlash.setOnClickListener(v -> toggleFlash());
    }
    public void startCamera(){
        android.util.Size screenSize = new android.util.Size(2 * previewView.getWidth(), 2 * previewView.getHeight());
        ListenableFuture<ProcessCameraProvider> listenableFuture = ProcessCameraProvider.getInstance(this);

        listenableFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = listenableFuture.get();

                Preview preview = new Preview.Builder().build();

                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(lensFacing)
                        .build();

                ImageCapture imageCapture = new ImageCapture.Builder().build();

                // Chọn độ phân giải cho camera
                ResolutionSelector resolutionSelector = new ResolutionSelector.Builder()
                        .setResolutionStrategy(new ResolutionStrategy(screenSize,
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                        .build();

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setResolutionSelector(resolutionSelector)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build();

                imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(previewView.getContext()), this::processImage);

                cameraProvider.unbindAll();

                camera = cameraProvider.bindToLifecycle(MainActivity.this, cameraSelector, preview, imageCapture, imageAnalysis);
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
            } catch (ExecutionException | InterruptedException e) {
                throw new RuntimeException(e);
            }

        }, ContextCompat.getMainExecutor(this));
    }
    public void switchCamera() {
        lensFacing = (lensFacing == CameraSelector.LENS_FACING_BACK) ?
                CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK;
        startCamera();
    }
    public void toggleFlash() {
        if (camera != null && camera.getCameraInfo().hasFlashUnit()) {
            boolean isFlashOn = (camera.getCameraInfo().getTorchState().getValue() == TorchState.ON);
            camera.getCameraControl().enableTorch(!isFlashOn);
        }
    }
    private Mat imageProxyToMat(ImageProxy image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        Mat mat = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC1);
        mat.put(0, 0, bytes);
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_YUV2BGR_I420);
        return mat;
    }

    private void loadOpenCV() {
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV loaded failed!");
            Toast.makeText(this, "OpenCV was not found!", Toast.LENGTH_LONG).show();
        }
    }
    private void loadYunetModel() {
        byte[] buffer;
        try {
            InputStream is = this.getResources().openRawResource(R.raw.face_detection_yunet_2023mar);
            int size = is.available();
            buffer = new byte[size];
            int bytesRead = is.read(buffer);
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Yunet model loaded failed" + e);
            Toast.makeText(this, "Yunet model was not found", Toast.LENGTH_LONG).show();
            return;
        }

        MatOfByte mModelBuffer = new MatOfByte(buffer);
        MatOfByte mConfigBuffer = new MatOfByte();
        faceDetector = FaceDetectorYN.create("onnx", mModelBuffer, mConfigBuffer, new Size(320, 320));
        Log.i(TAG, "FaceDetectorYN initialized successfully!");
    }

    private void processImage(ImageProxy imageProxy) {
        Mat mat = imageProxyToMat(imageProxy);
        Mat faces = new Mat();

        if (mInputSize == null) {
            mInputSize = new Size(Math.round(mat.cols()/mScale), Math.round(mat.rows()/mScale));
            faceDetector.setInputSize(mInputSize);
        }

        // Chuyển đổi sang BGR trước khi detect khuôn mặt
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2BGR);
        Imgproc.resize(mat, mat, mInputSize);

        faceDetector.detect(mat, faces);
        visualize(mat, faces);

        runOnUiThread(() -> updatePreview(mat, previewView));

        mat.release();
        faces.release();
        imageProxy.close();
    }

    private void visualize(Mat frame, Mat faces) {
        int thickness = 2;
        float[] faceData = new float[faces.cols() * faces.channels()];

        for (int i = 0; i < faces.rows(); i++) {
            faces.get(i, 0, faceData);

            Log.d(TAG, "Detected face (" + faceData[0] + ", " + faceData[1] + ", " +
                    faceData[2] + ", " + faceData[3] + ")");

            Imgproc.rectangle(frame, new Rect(Math.round(mScale * faceData[0]), Math.round(mScale * faceData[1]),
                            Math.round(mScale * faceData[2]), Math.round(mScale * faceData[3])),
                    new Scalar(0, 255, 0), thickness);
            Imgproc.circle(frame, new Point(Math.round(mScale * faceData[4]), Math.round(mScale * faceData[5])),
                    2, new Scalar(255, 0, 0), thickness);
            Imgproc.circle(frame, new Point(Math.round(mScale * faceData[6]), Math.round(mScale * faceData[7])),
                    2, new Scalar(0, 0, 255), thickness);
            Imgproc.circle(frame, new Point(Math.round(mScale * faceData[8]), Math.round(mScale * faceData[9])),
                    2, new Scalar(0, 255, 255), thickness);
            Imgproc.circle(frame, new Point(Math.round(mScale * faceData[10]), Math.round(mScale * faceData[11])),
                    2, new Scalar(255, 255, 0), thickness);
            Imgproc.circle(frame, new Point(Math.round(mScale * faceData[12]), Math.round(mScale * faceData[13])),
                    2, new Scalar(255, 0, 255), thickness);
        }
    }

    private void updatePreview(Mat frame, PreviewView previewView) {
        Bitmap bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame, bitmap);
        previewView.post(() -> previewView.setBackground(new BitmapDrawable(this.getResources(), bitmap)));
    }
}