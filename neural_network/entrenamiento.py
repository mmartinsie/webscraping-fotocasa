"""Forward pass, backpropagation and gradient-descent update.

Part of the obsolete from-scratch network; see ``README.md``.
"""

import numpy as np


def mse(Ypredich, Yreal):
    """Return ``(mean_squared_error, residual)`` for a batch of predictions."""
    residual = np.array(Ypredich) - np.array(Yreal)
    return np.mean(residual**2), residual


def entrenamiento(X, Y, red_neuronal, lr=0.05):
    """Run one training step over ``red_neuronal`` and return its output."""
    # Forward pass: output[i] is the activation of layer i (output[0] is the input).
    output = [X]
    for num_capa in range(len(red_neuronal)):
        z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
        a = red_neuronal[num_capa].funcion_act[0](z)
        output.append(a)

    # Backpropagation, from the output layer back to the first hidden layer.
    back = list(range(len(output) - 1))
    back.reverse()

    delta = []
    w_next = None  # transposed weights of the layer processed just before this one
    for layer_idx in back:
        a = output[layer_idx + 1]
        if layer_idx == back[0]:
            error = mse(a, Y)[1] * red_neuronal[layer_idx].funcion_act[1](a)
        else:
            error = delta[-1] @ w_next * red_neuronal[layer_idx].funcion_act[1](a)
        delta.append(error)

        w_next = red_neuronal[layer_idx].W.transpose()

        # Gradient descent update.
        red_neuronal[layer_idx].b = red_neuronal[layer_idx].b - np.mean(delta[-1], axis=0, keepdims=True) * lr
        red_neuronal[layer_idx].W = red_neuronal[layer_idx].W - output[layer_idx].transpose() @ delta[-1] * lr

    return output[-1]
