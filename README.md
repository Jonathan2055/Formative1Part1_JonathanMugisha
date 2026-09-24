# Formative 1 Part 1 — Neural Network from Scratch

A single-layer neural network (`Linear → activation → loss`) implemented from scratch in
NumPy, with its own forward pass, backward pass (backpropagation), and SGD optimizer —
no autograd, no deep learning framework.

## What's implemented

- **`nn/layers/linear.py`** — `Linear`: `z = xW + b`, Xavier-initialized weights, full
  forward/backward with in-place gradient buffers.
- **`nn/activations/`** — `ReLU`, `Sigmoid`, `Softmax`, each with a numerically stable
  forward pass (no overflow on large-magnitude inputs) and a matching backward pass.
- **`nn/losses/`** — `CrossEntropyLoss` (binary) and `CategoricalCrossEntropyLoss`
  (multi-class), both with clipped predictions to avoid `log(0)`.
- **`nn/optim/sgd.py`** — `SGD`: vanilla `param -= lr * grad`, applied to every parameter
  a module exposes via `parameters()`.
- **`main.py`** — wires a `Linear + Sigmoid + CrossEntropyLoss` pipeline together, trains
  it with SGD on a small AND-gate toy dataset, and reports loss/accuracy — the end-to-end
  sanity check that every piece works together, not just in isolation.

## Design notes

- Every module follows the same contract: `forward` computes and caches whatever
  `backward` will need; `backward` returns the gradient with respect to its input and
  writes parameter gradients **in place** (`self.dW[...] = ...`, not `self.dW = ...`), so
  an optimizer that captured a reference to a parameter's gradient array early still sees
  every update.
- Every stability-sensitive operation (softmax, sigmoid, both losses) is guarded against
  overflow or `log(0)` using the max-subtraction / clipping techniques covered in the
  assignment, verified directly by the numerical-stability tests.
- `Softmax.backward` and the optimizer's per-parameter loop are the only loops in `nn/` —
  both are explicitly allowed by the vectorization rubric; every other forward/backward
  pass is fully vectorized NumPy.

## Running it

```bash
pytest              # run the full test suite, chapter by chapter
ruff check nn/       # lint the library code
ruff check main.py   # lint the training script
python main.py       # train on the toy AND-gate dataset and print progress
```

## Status

All chapter tests (Stages 1–10) pass, including the end-to-end convergence test in
`test_stage10_training_converges.py`. The toy AND-gate pipeline in `main.py` is the
scaffold the real competition submission will build on, swapping in the actual dataset
in place of `toy_data()`.

**Note on Stage 7:** `tests/activations/test_stage7_softmax_backward.py` imports
`CategoricalCrossEntropyLoss` (its `test_chained_with_categorical_cross_entropy_is_a_minus_y`
check, group C — the `a - y` shortcut), which isn't implemented until Stage 8. Running
Stage 7's test file in isolation before Stage 8 exists will show 3/4 passing with an
`ImportError` on that one case; this is expected and resolves once
`nn/losses/categorical_cross_entropy_loss.py` is in place.