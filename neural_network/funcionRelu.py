"""Activation functions and their derivatives.

Each activation is a ``(function, derivative)`` tuple so the training loop can
call ``act[0](x)`` on the forward pass and ``act[1](x)`` on the backward pass.

Part of the obsolete from-scratch network; see ``README.md``.
"""

import numpy as np


def derivada_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


relu = (
    lambda x: x * (x > 0),
    lambda x: derivada_relu(x),
)


def derivada_idem(x):
    x[x < 0] = -1
    x[x == 0] = 0
    x[x > 0] = 1
    return x


idem = (
    lambda x: x,
    lambda x: derivada_idem(x),
)

sigmoid = (
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: x * (1 - x),
)
