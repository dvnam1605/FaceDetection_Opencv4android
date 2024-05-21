package app.cameraapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.ImageView;
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

import org.opencv.core.Core;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
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
    private ImageView overlayImageView;
    private FaceDetectorYN faceDetector;
    private Net faceRecognizer;
    int lensFacing = CameraSelector.LENS_FACING_BACK;
    Camera camera;

    float mScale = 1.0f;
    private Size mInputSize = null;
    private final ActivityResultLauncher<String> activityResultLauncher = registerForActivityResult(new ActivityResultContracts.RequestPermission(), o -> {
        if (ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        }
    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        capture = findViewById(R.id.captureButton);
        toggleFlash = findViewById(R.id.flashButton);
        switchCamera = findViewById(R.id.switchCameraButton);
        overlayImageView = findViewById(R.id.overlayImageView);

        boolean isOpenCVLoaded = loadOpenCV();
        boolean isYuNetModelLoaded = loadYuNetModel();
        boolean isMobileFaceNetModelLoaded = loadMobileFaceNetModel();

        if (isOpenCVLoaded && isYuNetModelLoaded && isMobileFaceNetModelLoaded) {
            Toast.makeText(this, "All Models initialized successfully!", Toast.LENGTH_LONG).show();
            Log.i(TAG, "Models initialized successfully!");
        } else {
            Toast.makeText(this, "Models initialization failed!", Toast.LENGTH_LONG).show();
            Log.e(TAG, "Models initialization failed!");
        }

        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            activityResultLauncher.launch(Manifest.permission.CAMERA);
        }

        switchCamera.setOnClickListener(v -> switchCamera());
        toggleFlash.setOnClickListener(v -> toggleFlash());
    }

    public void startCamera() {
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

                // Camera resolution
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

    private Mat yuvToRgb(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        // U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        Mat yuvMat = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
        yuvMat.put(0, 0, nv21);

        Mat rgbMat = new Mat();
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_I420);
        Core.rotate(rgbMat, rgbMat, Core.ROTATE_90_CLOCKWISE);
        return rgbMat;
    }



    private boolean loadOpenCV() {
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
            return true;
        } else {
            Log.e(TAG, "OpenCV loaded failed!");
            Toast.makeText(this, "OpenCV was not found!", Toast.LENGTH_LONG).show();
            return false;
        }
    }

    private boolean loadYuNetModel() {
        byte[] buffer;
        try {
            InputStream is = this.getResources().openRawResource(R.raw.face_detection_yunet_2023mar);
            int size = is.available();
            buffer = new byte[size];
            int bytesRead = is.read(buffer);
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "YuNet loaded failed" + e);
            Toast.makeText(this, "YuNet model was not found", Toast.LENGTH_LONG).show();
            return false;
        }

        MatOfByte mModelBuffer = new MatOfByte(buffer);
        MatOfByte mConfigBuffer = new MatOfByte();
        faceDetector = FaceDetectorYN.create("onnx", mModelBuffer, mConfigBuffer, new Size(320, 320));

        Log.i(TAG, "YuNet initialized successfully!");
        return true;
    }

    private boolean loadMobileFaceNetModel() {
        byte[] PBuffer;
        byte[] MBuffer;
        try {
            InputStream Pis = this.getResources().openRawResource(R.raw.mobilefacenet);
            int size = Pis.available();
            PBuffer = new byte[size];
            int PBytesRead = Pis.read(PBuffer);
            Pis.close();
            InputStream Mis = this.getResources().openRawResource(R.raw.mobilefacenet_caffemodel);
            size = Mis.available();
            MBuffer = new byte[size];
            int MBytesRead = Mis.read(MBuffer);
            Mis.close();
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "MobileFaceNet loaded failed" + e);
            Toast.makeText(this, "MobileFaceNet model was not found", Toast.LENGTH_LONG).show();
            return false;
        }
        MatOfByte bufferProto = new MatOfByte();
        MatOfByte bufferModel = new MatOfByte();
        faceRecognizer = Dnn.readNetFromCaffe(bufferProto, bufferModel);
        Log.i(TAG, "MobileFaceNet initialized successfully!");
        return true;
    }

    private void processImage(ImageProxy imageProxy) {
        Mat mat = yuvToRgb(imageProxy);

        if (mInputSize == null) {
            mInputSize = new Size(Math.round(mat.cols() / mScale), Math.round(mat.rows() / mScale));
            faceDetector.setInputSize(mInputSize);
        }

        // Resize mat to the input size of the model
        Imgproc.resize(mat, mat, mInputSize);

        // Convert color to BGR
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2BGR);

        Mat faces = new Mat();
        faceDetector.detect(mat, faces);

        // Create a transparent overlay
        Mat overlay = Mat.zeros(mat.size(), CvType.CV_8UC4);

        // Draw bounding boxes on the transparent overlay
        visualize(overlay, faces);

        // Update the overlay ImageView with the processed overlay
        updateOverlay(overlay);

        mat.release();
        faces.release();
        overlay.release();
        imageProxy.close();
    }

    // Draw bounding boxes on the transparent overlay
    private void visualize(Mat overlay, Mat faces) {
        int thickness = 1;
        float[] faceData = new float[faces.cols() * faces.channels()];

        for (int i = 0; i < faces.rows(); i++) {
            faces.get(i, 0, faceData);
            Imgproc.rectangle(overlay, new Rect(Math.round(mScale * faceData[0]), Math.round(mScale * faceData[1]),
                            Math.round(mScale * faceData[2]), Math.round(mScale * faceData[3])),
                    new Scalar(0, 255, 0, 255), thickness); // Using RGBA for transparency
        }
    }

    private void updateOverlay(Mat overlay) {
        Bitmap bitmap = Bitmap.createBitmap(overlay.cols(), overlay.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(overlay, bitmap);
        runOnUiThread(() -> overlayImageView.setImageBitmap(bitmap));
    }
}