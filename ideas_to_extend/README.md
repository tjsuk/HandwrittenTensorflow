# Ideas to Extend — Detailed Guides

Each file here expands one item from `handwritten_digit_recognition.ipynb`'s closing "Ideas to
extend" section into a full, self-contained walkthrough: why the idea matters, a step-by-step
explanation of how to do it, complete runnable code, and things to try afterward to deepen the
understanding. They assume you've already worked through the main notebook, and reference its
variables (`model`, `history`, `x_train`, `cnn_model`, etc.) directly.

| Guide | Extends | What you'll learn |
|---|---|---|
| [01_more_epochs_and_layers.md](01_more_epochs_and_layers.md) | Steps 5 and 7 | Training for longer and adding a Dense layer, with a direct loss-curve comparison |
| [02_try_the_cnn.md](02_try_the_cnn.md) | Bonus: Try a CNN yourself | Widening the bonus CNN, and making it the notebook's main model by changing Steps 4, 5, and 11b |
| [03_test_unusual_handwriting.md](03_test_unusual_handwriting.md) | Step 11 | Pushing the drawing canvas with unusual handwriting styles, and using the "teach the model" feature to correct and observe catastrophic forgetting |
| [04_explore_backpropagation.md](04_explore_backpropagation.md) | Bonus: Watch backpropagation happen | Turning the single-pixel gradient inspection into a full gradient-magnitude heatmap across the whole image |
| [05_compare_with_pytorch.md](05_compare_with_pytorch.md) | Summary | Running the companion PyTorch notebook side by side and checking its comparison table's claims for real |

## Suggested order

If you want to work through all five, this order builds on itself reasonably well:

1. **03** (unusual handwriting) — quick, no new dependencies, just uses the canvas you already have
2. **01** (more epochs / layers) — extends the training loop you already know
3. **04** (backpropagation) — deepens the same bonus section you've already run once
4. **02** (the CNN) — a bigger architectural change, best tackled once training itself feels familiar
5. **05** (compare with PyTorch) — introduces a whole second repository, so do it last
