# Handwritten Digit Recognition with TensorFlow

A beginner-friendly Jupyter notebook that trains a neural network to recognise
handwritten digits (0-9) using the [MNIST dataset](https://systemds.apache.org/datasets/mnist)
and TensorFlow/Keras. It's written as a self-contained **learning exercise**, not just
a script to run — every step includes plain-English explanations of *what* the code
does and *why*, what results to expect, and what to do if something looks off.
It also includes a drawing canvas so you can test (and even correct) the model on
your own handwriting.

This is the TensorFlow counterpart to
[HandwrittenPyTorch](https://github.com/tjsuk/HandwrittenPyTorch), a separate project solving the
same problem with PyTorch. The two aren't required reading for each other, but if you're comparing
the two frameworks, they're a good pair to read together — see
[`ideas_to_extend/05_compare_with_pytorch.md`](ideas_to_extend/05_compare_with_pytorch.md) for a
guided, side-by-side comparison.

**Part of a series** on machine learning fundamentals, each solving a related problem at a
different level of the stack:

1. **[NumPy_Fundamentals](https://github.com/tjsuk/ExplainingNumPy)** — build a neural network
   from scratch using nothing but NumPy
2. **[Scikit_Learn_Guide](https://github.com/tjsuk/Scikit_Learn_Guide)** — a handwritten-digit
   classifier built with scikit-learn's toolkit
3. **HandwrittenTensorflow** (this notebook) — a full digit-recognition project built with
   TensorFlow/Keras
4. **[HandwrittenPyTorch](https://github.com/tjsuk/HandwrittenPyTorch)** — the same
   digit-recognition project built with PyTorch

They're independent and don't require reading in order, but each links back to the others where
the connection is most relevant.

## What's inside

The notebook walks through the full workflow of a machine learning project, step by step:

1. **Import libraries** — TensorFlow, matplotlib, numpy
2. **Load the MNIST dataset** — 70,000 labelled handwritten digit images
3. **View 20 random handwritten digits** — get a feel for the data before doing anything else
4. **Prepare the data** — normalise pixel values so the model trains efficiently
5. **Build the model** — a simple `Sequential` neural network, with an explanation of every layer and what the model summary output means
6. **Compile the model** — configure the optimizer, loss function, and metrics, explained in plain English
7. **Train the model** — with an explanation of what happens during training, and an on-screen reminder to wait for it to finish before continuing
8. **Evaluate the model** — understand what the test accuracy score means, why it might be lower than expected, and how to improve it
9. **Make predictions** — see exactly what image and array of numbers the model works with, plus a bar-chart view of the model's confidence across several random examples
10. **Visualise predictions** — compare predicted vs actual digits at a glance
11. **Draw your own digit** — an interactive canvas where you draw a digit, get a live prediction, confirm whether it was right, and optionally teach the model from a mistake (a simple demo of *online learning*)
12. **Save the trained model** — with an explanation of what a `.keras` file actually contains

It finishes with a **Summary** recapping the whole workflow and key terms (loss, accuracy,
overfitting, normalisation, etc.), followed by two bonus sections:

- **Watch backpropagation happen** — un-hides what `model.fit()` normally does invisibly. A
  minimal one-weight toy example computes a gradient with `tf.GradientTape()` and checks it
  against a hand-worked calculus derivative (they match exactly), then the same technique is
  applied to the real trained model — inspecting actual gradient values on a real batch, and
  applying one genuine optimizer step to watch a specific weight change by a real, measurable
  amount (its original weights are restored immediately afterwards, so the demo has no side
  effects on the rest of the notebook).
- **Try a CNN yourself** — what a Convolutional Neural Network (CNN) is, how/why it differs from
  the simple network built here, exactly which cells above you'd need to change to convert it to
  a CNN, and a runnable CNN cell you can use to compare its accuracy directly against the original
  model (no need to edit anything above it).

## Requirements

- **Python 3.9–3.12** (TensorFlow does not yet support Python 3.13+)
- pip

## Setup

1. **Clone or download this repository.**

2. **Create a virtual environment** in the project folder:

   ```bash
   python -m venv .venv
   ```

3. **Activate it.**

   Windows (PowerShell/cmd):
   ```bash
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

4. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Register the environment as a Jupyter kernel** (so the notebook can find your installed packages):

   ```bash
   python -m ipykernel install --user --name=handwritten-digit-venv --display-name "Python (handwritten-digit-venv)"
   ```

## Running the notebook

Launch Jupyter Lab **using the venv's own executable** rather than relying on `activate` alone
(if another Python install is earlier on your `PATH`, a plain `jupyter lab` command can silently
launch the wrong environment — see the Troubleshooting section below):

```bash
.venv\Scripts\jupyter-lab.exe
```

(macOS/Linux: `.venv/bin/jupyter-lab`, after activating the venv.)

Open `handwritten_digit_recognition.ipynb`, and make sure it's using the **"Python
(handwritten-digit-venv)"** kernel (in VS Code: click the kernel picker in the top-right of the
notebook; in Jupyter Lab: use the **Kernel → Change Kernel** menu).

Then run the cells from top to bottom, reading the explanations as you go. Training (Step 7)
takes well under a minute on a typical CPU and should reach around 97–98% test accuracy
(Step 8) — the notebook explains what that number means and what to check if yours looks
noticeably lower.

### Trying the drawing canvas (Step 11)

After the model is trained, Step 11 has a canvas widget you can draw on:

1. Draw a single digit with your mouse (a large, thick stroke works best)
2. Click **Predict** to see what the model thinks it is, along with its confidence
3. Confirm whether it was right:
   - **Yes, correct** — nothing more to do
   - **No, wrong** — pick the actual digit from the dropdown and click **Teach the model**. This
     runs a quick training step on your drawing so the model learns from the mistake. It's a
     simple demo of *online learning* — don't rely on it too heavily, since repeatedly fine-tuning
     on a handful of examples can make the model overfit to your handwriting and forget some of
     what it learned from MNIST.
4. Click **Clear** to try another digit

This uses `ipycanvas` and `ipywidgets`, which are included in `requirements.txt`.

## Troubleshooting

**Widgets render as plain text** (e.g. `Canvas(height=280, ...)` instead of an actual canvas,
or `HBox(children=(...))` instead of real buttons):

This means the Jupyter **server** you're connected to isn't the one with `ipywidgets`/`ipycanvas`
installed — the widget-rendering extension is loaded by whichever Python environment started the
server, not by the kernel you select afterwards. Common causes:

- You ran `jupyter lab` from a terminal where the venv wasn't actually activated, so it launched
  from a different (often system-wide) Python install instead. Use the venv's own executable
  directly, as shown above, to avoid any ambiguity.
- **A server was already running.** If you run `jupyter lab` again while a server for the same
  folder is still running elsewhere, Jupyter just opens a new browser tab connected to that
  *existing* server rather than starting a fresh one — so restarting your terminal alone won't
  fix a bad server. Check for other running servers with `jupyter server list`, and fully stop
  any stale ones (`Ctrl+C` in their terminal, or end the process) before starting a new one from
  the venv.

**The drawing area needs scrollbars to see fully:**

Jupyter Lab boxes very tall cell outputs into a small scrollable region by default. Right-click
the output and choose **"Disable Scrolling for Outputs"**, or click the thin vertical bar running
down the output's left edge to toggle it between boxed and full-height.

## Notes

- TensorFlow on native Windows runs on CPU only (no GPU support since TensorFlow 2.11). This is
  fine for this project — MNIST is small and trains quickly on CPU. If you want GPU acceleration,
  look into WSL2 or the `tensorflow-directml-plugin` package.
- The saved model file (`handwritten_digit_model.keras`) is created when you run the "Save the
  trained model" step (Step 12) — it isn't included in this repository.

## Ideas to extend

- Train for more epochs, or add more Dense layers, and see how that affects both accuracy and
  training time
- Try the Convolutional Neural Network in the Bonus section, or make it the notebook's main model
  instead — see exactly which cells to change
- Test the model against unusual or messy handwriting using the drawing canvas's "teach the
  model" feature
- Explore the backpropagation bonus section further by inspecting gradients at different pixel
  indices, or across the whole image at once
- Compare this notebook side-by-side with the companion PyTorch version

Each of these has a detailed, step-by-step walkthrough with full explanations and runnable code
in [`ideas_to_extend/`](ideas_to_extend/README.md).
