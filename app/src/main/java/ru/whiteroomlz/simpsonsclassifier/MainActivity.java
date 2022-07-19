package ru.whiteroomlz.simpsonsclassifier;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.StringJoiner;

import okhttp3.Call;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends Activity {
    TextView labelsTextView;
    ImageView imageView;
    Button importImageButton;

    Handler handler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        labelsTextView = findViewById(R.id.labelsTextView);
        imageView = findViewById(R.id.imageView);
        importImageButton = findViewById(R.id.importImageButton);

        importImageButton.setOnClickListener(view -> {
            Intent imagePickerIntent =
                    new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(imagePickerIntent, 3);
        });

        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.ACCESS_NETWORK_STATE,
                    Manifest.permission.INTERNET
            }, 1);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            imageView.setImageURI(selectedImage);
            if (imageView.getVisibility() != View.VISIBLE) {
                imageView.setVisibility(View.VISIBLE);
            }

            Bitmap bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();
            AsyncTask.execute(() -> {
                try {
                    postRequest(bitmap);
                } catch (IOException | JSONException e) {
                    e.printStackTrace();
                }
            });
        }
    }

    void postRequest(Bitmap bitmap) throws IOException, JSONException {
        String postUrl = "http://10.0.2.2:5000/predict";

        String postBodyText = getStringImage(bitmap);
        MediaType mediaType = MediaType.parse("text/plain; charset=utf-8");
        RequestBody postBody = RequestBody.create(mediaType, postBodyText);

        OkHttpClient client = new OkHttpClient();

        Request request = new Request.Builder()
                .url(postUrl)
                .post(postBody)
                .build();
        Call call = client.newCall(request);
        Response response = call.execute();

        if (response.isSuccessful()) {
            assert response.body() != null;
            JSONObject object = new JSONObject(response.body().string());

            String label = object.getString("class_name");
            String[] labelParts = label.split("_");

            StringBuilder builder = new StringBuilder("");
            for (String labelPart : labelParts) {
                builder.append(labelPart.substring(0, 1).toUpperCase());
                builder.append(labelPart.substring(1));
                builder.append(" ");
            }

            handler.post(() -> labelsTextView.setText(builder.toString()));
        }
    }

    public String getStringImage(Bitmap bmp) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 100, baos);
        byte[] imageBytes = baos.toByteArray();
        return Base64.encodeToString(imageBytes, Base64.DEFAULT);
    }
}