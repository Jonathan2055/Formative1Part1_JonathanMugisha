"""ReLU activation: zeroes out negative values."""
import numpy as np

from nn.module import Module


class ReLU(Module):
    """ReLU activation, applied elementwise: max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute ReLU elementwise, and remember the mask.

        Args:
            x (np.ndarray): input, any shape.

        Returns:
            np.ndarray: max(0, x), elementwise, same shape as x.

        Sets:
            self.relu_mask (np.ndarray): boolean array, same shape as x,
                True where x > 0. Used by backward to zero out
                gradient at positions where the input was not
                strictly positive.
        """
        self.relu_mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, same shape as
                the original input to forward.

        Returns:
            np.ndarray: gradient of the loss with respect to
                this layer's input, same shape as grad_output.
        """
        return grad_output * self.relu_mask
