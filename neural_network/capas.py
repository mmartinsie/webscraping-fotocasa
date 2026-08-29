"""Single dense layer of the from-scratch network.

Part of the obsolete from-scratch network; see ``README.md``.
"""

import numpy as np
from scipy import stats


class capa:
    """A fully connected layer: weights ``W``, bias ``b`` and an activation.

    Weights and biases are drawn from a truncated normal on ``[0, 1]`` and scaled
    up (x100 / x1000) so the untrained network produces non-trivial outputs.
    """

    def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
        self.funcion_act = funcion_act
        self.b = (
            np.round(stats.truncnorm.rvs(0, 1, loc=0, scale=1, size=n_neuronas).reshape(1, n_neuronas), 3)
            * 1000
        )
        self.W = (
            np.round(
                stats.truncnorm.rvs(0, 1, loc=0, scale=1, size=n_neuronas * n_neuronas_capa_anterior).reshape(
                    n_neuronas_capa_anterior, n_neuronas
                ),
                3,
            )
            * 100
        )
