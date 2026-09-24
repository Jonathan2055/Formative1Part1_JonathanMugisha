"""Categorical cross-entropy loss, for one-hot multi-class targets."""
import numpy as np


class CategoricalCrossEntropyLoss:
    """Categorical cross-entropy loss over C classes.

    Does not subclass Module -- see the note in Chapter 6.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average categorical cross-entropy loss.

        Args:
            predictions (np.ndarray): softmax probabilities,
                shape (m, C). Clip away from exactly 0 before
                use -- see "The same clipping requirement as
                Chapter 6" above.
            targets (np.ndarray): one-hot true labels, shape
                (m, C).

        Returns:
            float: the scalar loss, averaged over the batch.

        Sets:
            self.a (np.ndarray): the clipped predictions, saved so
                backward reuses the identical clipped values
                rather than reclipping (or forgetting to).
            self.y (np.ndarray): the targets, saved unchanged so
                backward can use them.
        """
        self.a = np.clip(predictions, 1e-12, 1 - 1e-12)
        self.y = targets
        m = self.a.shape[0]
        loss = -np.sum(self.y * np.log(self.a))
        return float(loss / m)

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
            np.ndarray: dL/da, shape (m, C), same shape as the
                predictions passed to forward. Use the same
                clipped predictions here as in forward.
        """
        m = self.a.shape[0]
        return -(1 / m) * (self.y / self.a)
