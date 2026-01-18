from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
import time


class BDPVectorized:
    def __init__(self, K: int, latest_alpha: np.ndarray | None = None):
        self.K = K
        self.NUM_PAIRS = K * (K - 1) // 2
        self.key = jr.key(int(time.time_ns()))
        self.i_all, self.j_all = jnp.triu_indices(K, k=1)

        if latest_alpha is not None:
            self.alpha_t = jnp.array(latest_alpha)
        else:
            self.alpha_t = jnp.ones(K)

        self.frequency = jnp.zeros(self.K)

    def get_alphas(self) -> np.ndarray:
        return np.array(self.alpha_t)

    def submit_comparison(self, i: int, j: int, winner: int):
        Y_ij = 1 if winner == i else -1
        self.alpha_t = BDPVectorized.MM(self.alpha_t, i, j, Y_ij)

    def get_next_pair(self, temp: float = 1.0) -> Tuple[int, int]:
        pair_frequency = self.frequency[self.i_all] + self.frequency[self.j_all]
        distribution = BDPVectorized.softmax(-pair_frequency, temp)
        self.key, subkey = jr.split(self.key)

        next_idx = jr.choice(subkey, self.NUM_PAIRS, p=distribution)
        next_i: int = self.i_all[next_idx].astype(int)
        next_j: int = self.j_all[next_idx].astype(int)
        self.frequency = self.frequency.at[next_i].add(1)
        self.frequency = self.frequency.at[next_j].add(1)

        return next_i, next_j

    @staticmethod
    @jit
    def MM(alpha_t: jnp.ndarray, i: int, j: int, Y_ij: int) -> jnp.ndarray:
        alpha_0 = jnp.sum(alpha_t)

        C = alpha_t / alpha_0
        C_ij_denom = alpha_0 * (alpha_t[i] + alpha_t[j] + 1.0)
        C = C.at[i].set(
            ((alpha_t[i] + (1.0 + Y_ij) / 2.0) * (alpha_t[i] + alpha_t[j])) / C_ij_denom
        )
        C = C.at[j].set(
            ((alpha_t[j] + (1.0 - Y_ij) / 2.0) * (alpha_t[i] + alpha_t[j])) / C_ij_denom
        )

        D_ij_denom = alpha_0 * (alpha_0 + 1.0) * (alpha_t[i] + alpha_t[j] + 2.0)
        D_i = (
            (alpha_t[i] + (1.0 + Y_ij) / 2.0)
            * (alpha_t[i] + (3.0 + Y_ij) / 2.0)
            * (alpha_t[i] + alpha_t[j])
        ) / D_ij_denom
        D_j = (
            (alpha_t[j] + (1.0 - Y_ij) / 2.0)
            * (alpha_t[j] + (3.0 - Y_ij) / 2.0)
            * (alpha_t[i] + alpha_t[j])
        ) / D_ij_denom

        D_rest_denom = alpha_0 * (alpha_0 + 1.0)
        D_all = jnp.sum(alpha_t * (alpha_t + 1.0)) / D_rest_denom
        D_extra = (
            alpha_t[i] * (alpha_t[i] + 1.0) + alpha_t[j] * (alpha_t[j] + 1.0)
        ) / D_rest_denom
        D_rest = D_all - D_extra

        D = D_i + D_j + D_rest

        sum_ck_sq = jnp.sum(C**2)
        alpha_0_prime = (D - 1.0) / (sum_ck_sq - D)
        alpha_prime = C * alpha_0_prime

        return alpha_prime

    @staticmethod
    @jit
    def softmax(logits: jnp.ndarray, temp: float = 1.0) -> jnp.ndarray:
        scaled = logits / temp
        exped = jnp.exp(scaled - jnp.max(scaled, axis=-1, keepdims=True))
        normed = exped / jnp.sum(exped, axis=-1, keepdims=True)
        return normed
