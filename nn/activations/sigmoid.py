"""Sigmoid activation: squashes real values into (0, 1)."""
import numpy as np

from nn.module import Module


class Sigmoid(Module):
    """Sigmoid activation, applied elementwise: 1 / (1 + e^{-x})."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid elementwise, and remember the output.

        Args:
            x (np.ndarray): input, any shape.

        Returns:
            np.ndarray: sigmoid(x), elementwise, same shape as x.

        Sets:
            self.sigmoid_out (np.ndarray): the output of this forward call,
                same shape as x. backward reuses it directly via
                sigmoid_out(1 - sigmoid_out), instead of recomputing sigmoid from x.
        """
        z = np.where(x >= 0, -x, x)
        exp_z = np.exp(z)
        self.sigmoid_out = np.where(x >= 0, 1.0 / (1.0 + exp_z), exp_z / (1.0 + exp_z))
        return self.sigmoid_out

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
        return grad_output * self.sigmoid_out * (1 - self.sigmoid_out)
