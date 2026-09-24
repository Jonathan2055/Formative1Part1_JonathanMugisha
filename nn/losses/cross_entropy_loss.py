"""Binary cross-entropy loss."""
import numpy as np


class CrossEntropyLoss:
    """Binary cross-entropy loss for a single output probability.

    Does not subclass Module -- see the note below.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average binary cross-entropy loss.

        Args:
            predictions (np.ndarray): predicted probabilities,
                shape (m,) or (m, 1). Clip away from exactly
                0 or 1 before use -- see "Numerical stability"
                above.
            targets (np.ndarray): true labels, same shape as
                predictions, values 0 or 1.

        Returns:
            float: the scalar loss, averaged over the batch.

        Sets:
            self.a (np.ndarray): the clipped predictions, saved
                so backward reuses the identical clipped values
                rather than reclipping (or forgetting to).
            self.y (np.ndarray): the targets, saved unchanged so
                backward can use them.
        """
        self.a = np.clip(predictions, 1e-12, 1 - 1e-12)
        self.y = targets
        m = self.a.shape[0]
        loss = -np.sum(self.y * np.log(self.a) + (1 - self.y) * np.log(1 - self.a))
        return float(loss / m)

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
            np.ndarray: dL/da, same shape as the predictions
                passed to forward. Use the same clipped
                predictions here as in forward -- see
                "Numerical stability" above.
        """
        m = self.a.shape[0]
        grad_predictions = -(1 / m) * (self.y / self.a - (1 - self.y) / (1 - self.a))
        return grad_predictions
