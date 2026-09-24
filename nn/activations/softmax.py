"""Softmax activation: converts logits into a probability distribution."""
import numpy as np

from nn.module import Module


class Softmax(Module):
    """Softmax activation, applied row-wise to a batch of logits.

    Unlike ReLU or Sigmoid, each output depends on every logit in
    its own row rather than just the matching input, since every
    output shares the same per-row normalizing sum in its
    denominator.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for a batch of logits.

        Args:
            x (np.ndarray): logits, shape (batch_size, C).

        Returns:
            np.ndarray: probabilities, shape (batch_size, C).
            Each row sums to 1.

        Sets:
            self.a (np.ndarray): the output of this forward call,
                same shape as x. backward reuses it to build each
                example's Jacobian.
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        self.a = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self.a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, shape (m, C).

        Returns:
            np.ndarray: gradient of the loss with respect to
                this layer's input (the logits), shape (m, C).
        """
        m, C = grad_output.shape
        dz = np.zeros((m, C))
        for i in range(m):
            a_i = self.a[i]
            g_i = grad_output[i]
            dot = np.sum(g_i * a_i)
            dz[i] = a_i * (g_i - dot)
        return dz
