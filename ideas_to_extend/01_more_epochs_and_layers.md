# Extending Steps 5 and 7: Training for Longer, or Adding More Layers

## What you're extending

[Step 7](../handwritten_digit_recognition.ipynb) trains for a fixed `epochs=5` using the model
built in Step 5 (`Flatten` → `Dense(128, relu)` → `Dropout(0.2)` → `Dense(10, softmax)`), and
never questions either choice. This guide reruns training with both changed, so you can see the
effect directly instead of taking the defaults on faith.

## Why this matters

Two of the most impactful choices in any training setup are "for how long" and "how big a model"
— and both are cheap to experiment with here, since the whole notebook trains in well under a
minute:

- **More epochs** generally keeps reducing the training loss, but test accuracy (Step 8)
  eventually plateaus or even gets slightly worse once the model starts fitting quirks of the
  training set rather than genuinely useful patterns.
- **More layers/neurons** gives the model more capacity to learn complex patterns, but also more
  ways to overfit — which is exactly why Step 5 already includes a `Dropout` layer.

## How to do it

### 1. Train for more epochs

In the Step 7 code cell, change:

```python
history = model.fit(x_train, y_train, epochs=5)
```

to:

```python
history = model.fit(x_train, y_train, epochs=15)
```

If you just re-run the Step 7 cell as-is, it *continues* training the same `model` for 15 more
epochs on top of the 5 it already did. For a fair, from-scratch comparison instead, re-run Step 5
first (rebuilds `model` with fresh random weights), then Step 6 (recompiles it), then the edited
Step 7.

### 2. Watch the loss and accuracy history that `.fit()` already records

Unlike PyTorch's hand-written training loop, Keras' `.fit()` already returns a `History` object
you can plot directly — no extra bookkeeping needed:

```python
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.title("Training over 15 epochs")
plt.show()
```

### 3. Add another Dense layer

Edit the Step 5 cell, inserting another block between the existing `Dense(128, ...)` and
`Dropout(0.2)` lines:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),   # <-- new layer
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

Re-run Step 5 (you'll see the new layer and a higher parameter count in `model.summary()`), then
re-run Steps 6-8 to compile, train, and evaluate the new architecture from scratch.

## What to observe / think about

- **Diminishing returns**: the loss curve should fall quickly in the first few epochs, then
  flatten out. Compare Step 8's test accuracy after 5 epochs vs. after 15 — the gain is usually
  much smaller than the jump from 1 epoch to 5.
- **Parameter count**: `model.summary()` should now report more than the original 101,770 total
  parameters. More parameters isn't automatically "better" — watch whether training accuracy
  pulls ahead of test accuracy, which is Step 8's own warning sign for overfitting.
- This notebook doesn't hold out a separate validation set — Step 8's test set is only ever
  checked once, after training is finished. If you plan to compare many configurations like this,
  consider passing `validation_split=0.1` to `model.fit()` instead of repeatedly checking the real
  test set, so your final test accuracy stays an honest, unseen-data measurement.

## Related ideas worth knowing about

- **Learning rate schedules and other optimizers**: Step 6 uses `"adam"` with its default
  settings. Try `tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)` instead, and notice it
  typically needs more epochs to reach a comparable accuracy — that's why Adam has become such a
  common default.
- **`EarlyStopping`**: a Keras callback (`tf.keras.callbacks.EarlyStopping`) that automatically
  stops training once validation loss stops improving, instead of guessing a fixed epoch count up
  front.
