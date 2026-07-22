# Backpropagation From Scratch — Fitting a Sphere with NumPy

Part of a YouTube series on neural networks. We implement a tiny neural net in pure NumPy,
manually derive and compute every backprop gradient, then compare it to the same model
trained in PyTorch.

## What's inside

- Synthetic dataset: points on a sphere ($x^2+y^2+z^2=R^2$), 3D visualized.
- A 2 → 3 (tanh) → 1 network.
- **One full epoch by hand**: forward pass and backward pass, each shown algebraically
  (neuron by neuron) then in matrix form, ending with the actual weight update.
- Full NumPy training loop + fitted surface plot.
- Same model rebuilt and trained in PyTorch, for comparison.

## Notes

- Run cells in order — later sections reuse variables from earlier ones.
- The manual walkthrough uses sample #0; swap the index to try another point.
- PyTorch's `nn.MSELoss()` is scaled by `0.5` in the notebook to match the `0.5*(ŷ-y)²`
  loss used in the manual derivation, so the two loss curves compare directly.

## File

`BackpropFromScratch.ipynb` — pre-executed with all outputs and plots.