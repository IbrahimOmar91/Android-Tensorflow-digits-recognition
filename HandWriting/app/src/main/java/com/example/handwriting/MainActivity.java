package com.example.handwriting;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.Display;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;

import com.github.mikephil.charting.charts.HorizontalBarChart;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.github.mikephil.charting.interfaces.datasets.IBarDataSet;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;


//import android.support.v7.app.AppCompatActivity;

// Resources:
/*
 * This app is using exciting drawing view from this link:
 * https://www.youtube.com/watch?v=VbtMk0iPZgk
 *
 * For showing the results i used:
 * https://github.com/PhilJay/MPAndroidChart
 */

// Find Me
/*
 * Mail:     ibrahimomar357@gmail.com
 * GitHub:   https://github.com/IbrahimOmar91
 * LinkedIn: https://www.linkedin.com/in/ibrahimomar91/
 * Facebook: https://www.facebook.com/IbrahimM.Omar91
 */

public class MainActivity extends AppCompatActivity {

    private PaintView paintView;
    private Button btnClassify;
    private ImageButton btnClear;
    Interpreter tflite;
    private HorizontalBarChart chart;
    ArrayList<BarEntry> values = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        chart = findViewById(R.id.chart);
        chart.setDrawBarShadow(false);

        chart.setDrawValueAboveBar(true);

        chart.getDescription().setEnabled(false);


        chart.setMaxVisibleValueCount(11);

        XAxis xl = chart.getXAxis();
        xl.setPosition(XAxis.XAxisPosition.BOTTOM);
        xl.setDrawGridLines(false);
        xl.setCenterAxisLabels(false);
        xl.setTextSize(12);
        xl.setLabelCount(10);

        YAxis yl = chart.getAxisLeft();
        yl.setDrawAxisLine(true);
        yl.setDrawGridLines(true);
        yl.setAxisMinimum(0f);

        YAxis yr = chart.getAxisRight();
        yr.setDrawAxisLine(true);
        yr.setDrawGridLines(false);
        yr.setAxisMinimum(0f);

        chart.setFitBars(true);

        Legend l = chart.getLegend();
        l.setVerticalAlignment(Legend.LegendVerticalAlignment.BOTTOM);
        l.setHorizontalAlignment(Legend.LegendHorizontalAlignment.LEFT);
        l.setOrientation(Legend.LegendOrientation.HORIZONTAL);
        l.setDrawInside(false);
        l.setFormSize(8f);
        l.setXEntrySpace(4f);

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        btnClassify = findViewById(R.id.btnClassify);
        btnClassify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    paintView.setDrawingCacheEnabled(true);
                    paintView.buildDrawingCache();
                    Bitmap bitmap = paintView.getDrawingCache().copy(Bitmap.Config.RGB_565, false);
                    paintView.setDrawingCacheEnabled(false);
                    Bitmap resized = Bitmap.createScaledBitmap(bitmap, 28, 28, true);
                    ByteBuffer buff = bitmapToModelsMatchingByteBuffer(resized);
                    runInferenceOnFloatModel(buff);
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        });

        btnClear = findViewById(R.id.btnClear);
        btnClear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                paintView.clear();
            }
        });

        Display display = getWindowManager().getDefaultDisplay();
        Point size = new Point();
        display.getSize(size);
        int width = size.x;

        paintView = findViewById(R.id.paintView);
        paintView.setLayoutParams(new ConstraintLayout.LayoutParams(width, width));
        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metrics);
        paintView.init(metrics);

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater menuInflater = getMenuInflater();
        menuInflater.inflate(R.menu.main, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.normal:
                paintView.normal();
                return true;
            case R.id.emboss:
                paintView.emboss();
                return true;
            case R.id.blur:
                paintView.blur();
                return true;
        }

        return super.onOptionsItemSelected(item);
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("mnist_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @SuppressLint("DefaultLocale")
    private void runInferenceOnFloatModel(ByteBuffer byteBufferToClassify) {
        float[][] result = new float[1][10];
        tflite.run(byteBufferToClassify, result);

        values = new ArrayList<>();
        for (int i = 0; i < result[0].length; i++) {
            values.add(new BarEntry(i, result[0][i]*100.f));
        }

        BarDataSet set1;
        set1 = new BarDataSet(values, "Results in percentages");

        ArrayList<IBarDataSet> dataSets = new ArrayList<>();
        dataSets.add(set1);

        BarData data = new BarData(dataSets);
        data.setValueTextSize(12f);
        chart.setData(data);
        chart.animateY(400);
    }


    private ByteBuffer bitmapToModelsMatchingByteBuffer(Bitmap bitmap) {
        int SIZE = 28;
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(SIZE * SIZE * 4);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[SIZE * SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                int pixelVal = intValues[pixel++];
                for (float channelVal : pixelToChannelValue(pixelVal)) {
                    byteBuffer.putFloat(channelVal);
                }
            }
        }
        return byteBuffer;
    }


    private float[] pixelToChannelValue(int pixel) {
        float[] singleChannelVal = new float[1];
        float rChannel = (pixel >> 16) & 0xFF;
        float gChannel = (pixel >> 8) & 0xFF;
        float bChannel = (pixel) & 0xFF;
        singleChannelVal[0] = (rChannel + gChannel + bChannel) / 3 / 255.f;
        return singleChannelVal;
    }

}