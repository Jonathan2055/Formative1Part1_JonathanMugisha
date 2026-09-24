"""Train a single-layer network on a toy AND-gate dataset.

This assembles Linear, Sigmoid, and CrossEntropyLoss into a full
training loop, to confirm forward, backward, and the SGD optimizer
genuinely work together end to end before moving on to real data.
"""
import numpy as np

from nn.activations import Sigmoid
from nn.layers import Linear
from nn.losses import CrossEntropyLoss
from nn.optim import SGD

_trained_layer = None
_trained_activation = None
_trained_X = None
_trained_y = None


def toy_data() -> tuple[np.ndarray, np.ndarray]:
    """Build the toy AND-gate dataset used to sanity-check the pipeline.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y). X has shape (4, 2),
            the four AND-gate input pairs. y has shape (4, 1), the
            corresponding AND-gate outputs.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [0], [0], [1]], dtype=float)
    return X, y


def train(epochs: int = 4000, lr: float = 1.0, seed: int = 0) -> list[float]:
    """Train a Linear + Sigmoid model with binary cross-entropy on the toy data.

    Args:
        epochs (int): number of full-batch training iterations.
        lr (float): learning rate passed to the SGD optimizer.
        seed (int): seed for reproducible weight initialization.

    Returns:
        list[float]: the loss recorded at every epoch, in order.

    Sets:
        Module-level state -- the trained layer, activation, and the
        toy data used -- so accuracy() can evaluate this same model
        afterward without retraining.
    """
    global _trained_layer, _trained_activation, _trained_X, _trained_y

    np.random.seed(seed)
    X, y = toy_data()

    layer = Linear(in_features=2, out_features=1)
    activation = Sigmoid()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(layer.parameters(), lr=lr)

    loss_history = []
    for _ in range(epochs):
        z = layer.forward(X)
        a = activation.forward(z)
        loss = loss_fn.forward(a, y)
        loss_history.append(loss)

        grad = loss_fn.backward()
        grad = activation.backward(grad)
        layer.backward(grad)

        optimizer.step()
        optimizer.zero_grad()

    _trained_layer = layer
    _trained_activation = activation
    _trained_X = X
    _trained_y = y

    return loss_history


def accuracy(loss_history: list[float] = None) -> float:
    """Report classification accuracy of the trained model on the toy data.

    Args:
        loss_history (list[float]): unused; accepted so the function
            signature matches what the test imports. If no model has
            been trained yet in this session, train() is called first
            with its default arguments.

    Returns:
        float: fraction of toy-data examples classified correctly,
            between 0 and 1.
    """
    if _trained_layer is None:
        train()

    z = _trained_layer.forward(_trained_X)
    a = _trained_activation.forward(z)
    predictions = (a >= 0.5).astype(float)
    return float(np.mean(predictions == _trained_y))


if __name__ == "__main__":
    losses = train()
    for epoch in (0, 999, 1999, 2999, 3999):
        if epoch < len(losses):
            print(f"epoch {epoch}: loss = {losses[epoch]:.4f}")
    print(f"final accuracy: {accuracy():.2f}")
