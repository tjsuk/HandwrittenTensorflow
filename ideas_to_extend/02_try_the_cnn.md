# Extending the Bonus "Try a CNN Yourself" Section

## What you're extending

The **Bonus: Try a CNN yourself** section at the end of
[the notebook](../handwritten_digit_recognition.ipynb) already builds, trains, and evaluates a
`cnn_model` independently of the main `model`, printing its test accuracy next to Step 8's for
comparison. This guide goes further: tweaking the CNN's architecture, and making it the
notebook's *main* model instead of a side comparison.

## Why this matters

The original model (Step 5) flattens the image to 784 numbers immediately, throwing away all
information about which pixels are *near* each other. A Convolutional Neural Network (CNN) keeps
the image in its 2D grid shape for longer, using small filters that slide across it looking for
local patterns (edges, curves, loops) regardless of where in the image they appear. This is
usually a better match for image data, and typically reaches higher accuracy on MNIST than a
plain fully-connected network of a similar size.

## How to do it

### 1. Run the bonus cell as-is first

No setup needed beyond having already run Steps 1-4 (so `x_train`/`x_test`/`y_train`/`y_test`
exist) and Step 8 (so `test_accuracy` exists for the printed comparison). Just scroll down and run
the CNN bonus cell directly.

### 2. Tweak the architecture

The bonus `cnn_model` uses 32 and 64 filters in its two `Conv2D` layers. Try widening it:

```python
wider_cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

wider_cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
wider_cnn_model.fit(x_train_cnn, y_train, epochs=3)
```

Keras works out the `Flatten` layer's input size automatically from `Conv2D`/`MaxPooling2D`'s
output shape, so widening the filter counts needs no other changes — unlike PyTorch, where the
first `Linear` layer's input size has to be recalculated by hand.

### 3. Make the CNN the notebook's main model

Several cells need to change, since the CNN expects a different input shape than the original
model. Following the Bonus section's own list:

- **Step 4**: reshape the data once, right after normalising it:
  ```python
  x_train = x_train.reshape(-1, 28, 28, 1)
  x_test = x_test.reshape(-1, 28, 28, 1)
  ```
- **Step 5**: replace the `Sequential` model with the CNN architecture shown in the Bonus section
  (or your widened version from step 2 above).
- **Step 7, 8, 9**: no code changes needed — `model.fit(x_train, y_train, ...)`,
  `model.evaluate(x_test, y_test)`, and `model.predict(x_test)` all still work, since `x_train`
  and `x_test` are already reshaped.
- **Step 11b**: change `processed_image.reshape(1, 28, 28)` to
  `processed_image.reshape(1, 28, 28, 1)` in both `on_predict_clicked` and `on_teach_clicked`, so
  your drawings get the same extra channel dimension.

This is a nice contrast with the companion PyTorch notebook, where switching to a CNN only
requires changing Step 5 — `torchvision.datasets.MNIST` already hands back images shaped
`(1, 28, 28)`, so no separate reshape step is needed there.

## What to observe / think about

- **Parameter count**: `cnn_model.summary()` reports 121,930 parameters, more than the original
  model's 101,770, despite `Conv2D` layers sharing the same small filter across the whole image —
  most of the CNN's parameters actually live in its `Dense(64)` layer, not the convolutions
  themselves.
- **Accuracy**: the bonus section already prints both models' test accuracy side by side — the
  CNN should come out at least a little ahead, and often by a clearer margin once you widen it as
  in step 2 above.
- **Training time**: the notebook's own comment already warns the CNN cell takes longer than
  Step 7. Time both training loops (e.g. with Python's `time.time()` around `model.fit(...)`) and
  compare accuracy-per-second, not just final accuracy.
- **Where accuracy plateaus**: MNIST is a fairly easy dataset — both the original model and a
  small CNN comfortably exceed 97%. The gap between them tends to widen much more on harder image
  datasets (e.g. `tf.keras.datasets.fashion_mnist` or `cifar10`), which is worth trying if you
  want to see a CNN's advantage more dramatically.

## Related ideas worth knowing about

- **Batch normalization** (`tf.keras.layers.BatchNormalization`, inserted after each
  convolution): often speeds up and stabilises CNN training, and is standard practice in almost
  every modern CNN architecture.
- **Data augmentation**: randomly rotating, shifting, or zooming training images slightly (via
  `tf.keras.layers.RandomRotation`, `RandomZoom`, etc., added as layers at the start of the model)
  exposes the model to more variation than the raw dataset provides, which tends to help most on
  harder datasets than MNIST.
- **A third convolutional block**: adding a `Conv2D`/`MaxPooling2D` pair before flattening lets
  the network build up even more abstract features, though on an image as small as 28x28 you'll
  run out of spatial room to pool further fairly quickly.
