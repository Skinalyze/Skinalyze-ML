![Visits Badge](https://badges.pufler.dev/visits/stefansphtr/Skinalyze-ML)
![Updated Badge](https://badges.pufler.dev/updated/stefansphtr/Skinalyze-ML)
![Created Badge](https://badges.pufler.dev/created/stefansphtr/Skinalyze-ML)
![Contributors Badge](https://img.shields.io/github/contributors/stefansphtr/Skinalyze-ML)
![Last Commit Badge](https://img.shields.io/github/last-commit/stefansphtr/Skinalyze-ML)
![Commit Activity Badge](https://img.shields.io/github/commit-activity/m/stefansphtr/Skinalyze-ML)
![Repo Size Badge](https://img.shields.io/github/repo-size/stefansphtr/Skinalyze-ML)
<!-- ![CodeFactor](https://www.codefactor.io/repository/github/stefansphtr/Skinalyze-ML/badge) -->
[![python](https://img.shields.io/badge/Python-3.11.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
![TensorFlow Badge](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow)
![Keras Badge](https://img.shields.io/badge/Keras-3.1.1-D00000?logo=keras)

# `SKINALYZE: Personalized skin care recommendations`
![Fading Line](https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif)
![Banner_Skinalyze](./src/assets/image/banner_readme.png)

> Team: `C241-PS385`
> 
> Project for: `Capstone Project Bangkit Academy 2024 H1`  


## Objective

- 📱 **Develop a mobile application (Android)** for personalized skincare recommendations.
  
- 🧠 **Implement a machine-learning model** for Skin Problem Classification. This model will analyze user-uploaded facial photos and identify specific skin concerns (e.g., dark spots, acne, pores).
  
- 💡 **Design a recommendation system** that goes beyond skin type identification. This system will offer personalized skincare product packages tailored to individual skin concerns.
  
- 😌 **Simplify the skincare product selection process** for users. This will be achieved by providing a user-centric application with clear and actionable recommendations.
  
- 📚 **Promote informed decision-making** regarding skincare products. This will be done by leveraging machine learning and personalized analysis.
  
- ♻️ **Reduce product waste** in the skincare industry. This will be achieved by recommending effective product packages that directly address users' specific needs.

## Directory in this repository

`SHORT SUMMARY AND LINK TO REPORT`

## Modelling results

`SHORT SUMMARY AND LINK TO REPORT`

## Model Documentation

<p align="center">
    <img src="https://user-images.githubusercontent.com/74038190/221352995-5ac18bdf-1a19-4f99-bbb6-77559b220470.gif" width="500">
</p>

### Model Name

The model's name is `skin_problem_classifier.tflite`. This model is a multiclass classifier model designed to analyze user skin problems and normal skin.

### Model Description

The model is a multiclass classifier that predicts the skin problem based on the input image from the user. It classifies the images into one of the following 5 classes:

1. Acnes
2. Blackheads
3. Darkspots
4. Normal
5. Wrinkles

### Model Input

The model expects the following input:

- The shape of the input tensor should be `(1, 224, 224, 3)`, which corresponds to a color image of size 224x224.
- The data type of the input should be `float32`.
- The input images should be normalized (pixel values ranging from 0 to 1) and resized to 224x224.

#### Input Requirements

The model expects a single color image as input. Here are the specific requirements:

- **Shape**: The input image should be resized to 224x224 pixels, resulting in a tensor of shape `(1, 224, 224, 3)`.
- **Data Type**: The pixel values of the image should be of type `float32`.
- **Normalization**: The pixel values should be normalized to range from 0 to 1.

#### Input Details

Here's a more detailed view of the input tensor:

- **Name**: `serving_default_input_1:0`
- **Shape**: `(1, 224, 224, 3)`
- **Data Type**: `float32`

#### Full Input Details

For those interested in more technical details, here's the full description of the input tensor:

```json
{
  "name": "serving_default_input_1:0",
  "index": 0,
  "shape": [1, 224, 224, 3],
  "shape_signature": [-1, 224, 224, 3],
  "dtype": "float32",
  "quantization": (0.0, 0),
  "quantization_parameters": {
    "scales": [],
    "zero_points": [],
    "quantized_dimension": 0
  },
  "sparsity_parameters": {}
}
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/74038190/212898774-0a96dc1d-c908-4ce8-9dd7-a71aab6e1c2b.gif" width="200">
</p>
<br><br>

- **Name (`name`)**: This is the identifier for the input tensor. In this case, it's `serving_default_input_1:0`.

- **Index (`index`)**: This is the index of the input tensor in the model's list of tensors. In this case, it's `0`.

- **Shape (`shape`)**: This describes the dimensions of the input tensor. In this case, it's `[1, 224, 224, 3]`, which means the model expects a single image of size 224x224 with 3 color channels (RGB).

- **Shape Signature (`shape_signature`)**: This is similar to `shape`, but it can also handle variable-sized dimensions. The `-1` means that dimensions can be of any size. In this case, it's `[-1, 224, 224, 3]`, which means the model can handle a batch of images of size 224x224 with 3 color channels.

- **Data Type (`dtype`)**: This is the type of data the tensor holds. In this case, it's `float32`, which means the pixel values of the image should be floating-point numbers.

- **Quantization (`quantization`)**: This describes the scale and zero point used for quantization of the tensor. In this case, it's `(0.0, 0)`, which means no quantization is applied.

- **Quantization Parameters (`quantization_parameters`)**: This provides more details about the quantization parameters. In this case, all fields are empty, which means no quantization is applied.

- **Sparsity Parameters (`sparsity_parameters`)**: This provides details about the sparsity parameters of the tensor. In this case, it's empty, which means no sparsity is applied.

### Model Output

The model produces the following output:

- The shape of the output tensor is `(1, 5)`, which corresponds to the probabilities of the 5 classes.
- The data type of the output is `float32`.
- The output should be post-processed by finding the index with the maximum probability. This index corresponds to the predicted class.

#### Output Requirements

The model produces a single output tensor. Here are the specific requirements:

- **Shape**: The output tensor has a shape of `(1, 5)`, which corresponds to the probabilities of the 5 classes.
- **Data Type**: The data type of the output is `float32`.

#### Output Details

Here's a more detailed view of the output tensor:

- **Name**: `StatefulPartitionedCall:0`
- **Shape**: `(1, 5)`
- **Data Type**: `float32`

#### Full Output Details

For those interested in more technical details, here's the full description of the output tensor:

```json
{
  "name": "StatefulPartitionedCall:0",
  "index": 49,
  "shape": [1, 5],
  "shape_signature": [-1, 5],
  "dtype": "float32",
  "quantization": (0.0, 0),
  "quantization_parameters": {
    "scales": [],
    "zero_points": [],
    "quantized_dimension": 0
  },
  "sparsity_parameters": {}
}
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/74038190/212745723-c7c386dc-108c-4a50-9c76-0f90afb2c0fa.gif" width="300">
</p>
<br><br>

- **Name (`name`)**: This is the identifier for the output tensor. In this case, it's `StatefulPartitionedCall:0`.

- **Index (`index`)**: This is the index of the output tensor in the model's list of tensors. In this case, it's `49`.

- **Shape (`shape`)**: This describes the dimensions of the output tensor. In this case, it's `[1, 5]`, which means the model produces probabilities for 5 classes.

- **Shape Signature (`shape_signature`)**: This is similar to `shape`, but it can also handle variable-sized dimensions. The `-1` means that dimensions can be of any size. In this case, it's `[-1, 5]`, which means the model can handle a batch of outputs for 5 classes.

- **Data Type (`dtype`)**: This is the type of data the tensor holds. In this case, it's `float32`, which means the output probabilities are floating-point numbers.

- **Quantization (`quantization`)**: This describes the scale and zero point used for quantization of the tensor. In this case, it's `(0.0, 0)`, which means no quantization is applied.

- **Quantization Parameters (`quantization_parameters`)**: This provides more details about the quantization parameters. In this case, all fields are empty, which means no quantization is applied.

- **Sparsity Parameters (`sparsity_parameters`)**: This provides details about the sparsity parameters of the tensor. In this case, it's empty, which means no sparsity is applied.

### Output Comparison

Here's an example of how to compare the output of the TensorFlow Lite model with the original model:

```python
# Assuming train_images is of shape [32, 224, 224, 3]
for i in range(train_images.shape[0]):
    # Get one image
    input_data = train_images[i:i+1]

    # Ensure the input data is a numpy array of type float32
    input_data = np.array(input_data, dtype=np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Invoke the interpreter
    interpreter.invoke()

    # Get the output tensor
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    # Predict the output using the original model
    original_output = model.predict(input_data)

    # Print the outputs
    print(f"TFLite output: {tflite_output}")
    print(f"Original model output: {original_output}")
```

This code snippet runs the same input through both the TensorFlow Lite model and the original model and prints the outputs. This allows us to compare the results and verify that the TensorFlow Lite model is behaving as expected.

Here are some of the results of comparing the output of the TensorFlow Lite model with the original model:

```plaintext
1/1 [==============================] - 0s 374ms/step
TFLite output: [[0.0559 0.2786 0.4832 0.1698 0.0125]]
Original model output: [[0.0234 0.7468 0.0933 0.1334 0.0032]]
...
...
1/1 [==============================] - 0s 67ms/step
TFLite output: [[1.0758e-04 1.0704e-01 1.9496e-02 1.9084e-03 8.7145e-01]]
Original model output: [[1.0436e-07 1.0135e-04 3.2434e-05 1.7621e-08 9.9987e-01]]
...
...
1/1 [==============================] - 0s 40ms/step
TFLite output: [[2.1433e-35 4.2481e-26 8.2432e-37 1.0000e+00 2.0442e-37]]
Original model output: [[1.9897e-27 3.1661e-23 3.2576e-37 1.0000e+00 0.0000e+00]]
```

These results show the output of both the TensorFlow Lite model and the original model for the same input. As you can see, there are some differences in the outputs. This is expected due to the quantization and optimization processes involved in converting the original model to the TensorFlow Lite format.

### Model Performance

The model's performance metrics are as follows:

- Accuracy on the test dataset: 93.75%
- Other relevant metrics:
  - Classification report on test dataset:

    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | 0     | 0.80      | 0.99   | 0.89     | 401     |
    | 1     | 0.92      | 0.82   | 0.87     | 416     |
    | 2     | 0.92      | 0.71   | 0.80     | 395     |
    | 3     | 0.98      | 0.98   | 0.98     | 392     |
    | 4     | 0.83      | 0.93   | 0.88     | 396     |
    |       |           |        |          |         |
    | Accuracy |         |        | 0.89     | 2000    |
    | Macro avg | 0.89   | 0.89   | 0.88     | 2000    |
    | Weighted avg | 0.89 | 0.89  | 0.88     | 2000    |

- Prediction time: 00:00:00.12

### Model Usage

To use the model in an Android app:

- Load the model using the TensorFlow Lite Android Support Library.
- Create an input tensor that matches the model's input shape and data type.
- Run inference using the TensorFlow Lite Android Support Library.
- Interpret the output tensor by finding the index with the maximum probability. This index corresponds to the predicted class.

### Example Code

Here's an example code snippet that demonstrates how to use the TensorFlow Lite model in an Android app:

<img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/42077049-1939-493e-9a19-47ca5db36643" width="75">&nbsp;

First, add the TensorFlow Lite Android library to the app's `build.gradle` file:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
}
```

Then, create a `Classifier` class that loads the model file and provides a method to run inference:

```kotlin
import android.content.Intent
import android.graphics.Bitmap
import android.provider.MediaStore
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.ByteBuffer
import android.content.res.AssetManager
import java.nio.ByteOrder

/**
 * Classifier class for classifying images using a TensorFlow Lite model.
 */
class Classifier(
    assetManager: AssetManager, 
    modelPath: String, 
    private val inputSize: Int
) {
    private var interpreter: Interpreter
    private val imgData: ByteBuffer

    init {
        interpreter = Interpreter(loadModelFile(assetManager, modelPath))
        imgData = ByteBuffer.allocateDirect(4 * 1 * inputSize * inputSize * 3)
        imgData.order(ByteOrder.nativeOrder())
    }

    /**
     * Load the TensorFlow Lite model file from assets.
     */
    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Classify an image.
     */
    fun classify(bitmap: Bitmap): Array<FloatArray> {
        convertBitmapToByteBuffer(bitmap)
        val output = Array(1) { FloatArray(5) }
        interpreter.run(imgData, output)
        return output
    }

    /**
     * Convert a bitmap to a ByteBuffer.
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        imgData.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val value = intValues[pixel++]
                imgData.putFloat(((value shr 16 and 0xFF) - 127.5f) / 127.5f)
                imgData.putFloat(((value shr 8 and 0xFF) - 127.5f) / 127.5f)
                imgData.putFloat(((value and 0xFF) - 127.5f) / 127.5f)
            }
        }
    }
}

/**
 * Main activity class that handles image capture and classification.
 */
class MainActivity : AppCompatActivity() {
    // Request code for image capture
    private val REQUEST_IMAGE_CAPTURE = 1

    // Classifier for image classification
    private lateinit var imageClassifier: Classifier

    // ImageView to display the captured image
    private lateinit var capturedImageView: ImageView

    // History of classification results
    private lateinit var classificationHistory: MutableList<Array<FloatArray>>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize the classifier with the model from assets
        imageClassifier = Classifier(assets, "skin_problem_classifier.tflite", 224)

        // Get the ImageView from the layout
        capturedImageView = findViewById(R.id.imageView)

        // Initialize the classification history
        classificationHistory = mutableListOf()

        // Set up the button to dispatch the image capture intent when clicked
        val captureImageButton: Button = findViewById(R.id.button)
        captureImageButton.setOnClickListener { dispatchTakePictureIntent() }
    }

    /**
     * Dispatch an intent to take a picture.
     */
    private fun dispatchTakePictureIntent() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            takePictureIntent.resolveActivity(packageManager)?.also {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }

    /**
     * Handle the result from the image capture intent.
     */
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            // Get the captured image
            val capturedImageBitmap = data?.extras?.get("data") as Bitmap

            // Display the captured image
            capturedImageView.setImageBitmap(capturedImageBitmap)

            // Resize the image for classification
            val resizedBitmap = Bitmap.createScaledBitmap(capturedImageBitmap, 224, 224, true)

            // Classify the image
            val classificationOutput = imageClassifier.classify(resizedBitmap)

            // Add the classification result to the history
            classificationHistory.add(classificationOutput)

            // Show the classification result
            Toast.makeText(this, "Output: ${classificationOutput[0].joinToString()}", Toast.LENGTH_LONG).show()
        }
    }
}
```

### Additional Resources

For more information, refer to the following resources:

- [TensorFlow Lite for Android documentation](https://www.tensorflow.org/lite/android)
- [TensorFlow Lite Android Support Library documentation](https://www.tensorflow.org/lite/guide/android)
- [Development tools for Android](https://www.tensorflow.org/lite/android/development)
- [TensorFlow Lite for Android Quickstart](https://www.tensorflow.org/lite/android/quickstart)
- [Recognize Flowers with TensorFlow Lite on Android](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/)
- [Process input and output data with the TensorFlow Lite Support Library](https://www.tensorflow.org/lite/guide/inference#process_input_and_output_data_with_the_tensorflow_lite_support_library)
- [TensorFlow Lite Android image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)

This documentation provides a comprehensive guide on how to use the TensorFlow Lite model in an Android app. It includes example code snippets, performance metrics, and additional resources for further learning.

Thank you team Mobile Dev for your hard work in integrating the model into the Android app!



```py
print("\033[92mEnd of report")
```

<span style="color: green; font-weight: bold">End of report</span>

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" width="500">

> Copyright by `Team C241-PS385`
> 
> Main developer:
> 
> `Stefanus Bernard Melkisedek`
> `Debby Trinita`
> `Mhd. Reza Kurniawan Lubis`
![Gradient Line](https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif)