from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
import time
from pydantic import BaseModel, field_validator, model_validator

FIELD_DTYPES = {
    "alpha_t": jnp.float32,
    "frequency": jnp.int32,
    "key": jnp.uint32,
}


class BDPVectorized(BaseModel):
    K: int
    alpha_t: jnp.ndarray
    frequency: jnp.ndarray
    key: jnp.ndarray

    class Config:
        json_encoders = {jnp.ndarray: lambda v: v.tolist()}
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    def initialize_missing(cls, values):
        K = values.get("K", 0)

        defaults = {
            "alpha_t": lambda: jnp.ones(K, dtype=jnp.float32),
            "frequency": lambda: jnp.zeros(K, dtype=jnp.int32),
            "key": lambda: jr.PRNGKey(int(time.time_ns())),
        }

        for field, default_fn in defaults.items():
            if not values.get(field):
                values[field] = default_fn()

        return values

    @field_validator(*FIELD_DTYPES.keys(), mode="before")
    def ensure_correct_dtype(cls, v, info):
        dtype = FIELD_DTYPES[info.field_name]
        return jnp.array(v, dtype=dtype)

    def get_alphas(self) -> np.ndarray:
        return np.array(self.alpha_t)

    def submit_comparison(self, i: int, j: int, winner: int):
        Y_ij = 1 if winner == i else -1
        self.alpha_t = BDPVectorized.MM(self.alpha_t, i, j, Y_ij)

    def get_next_pair(self, temp: float = 1.0) -> Tuple[int, int]:
        i_all, j_all = jnp.triu_indices(self.K, k=1)
        pair_frequency = self.frequency[i_all] + self.frequency[j_all]
        distribution = BDPVectorized.softmax(-pair_frequency, temp)
        self.key, subkey = jr.split(self.key)

        NUM_PAIRS = self.K * (self.K - 1) // 2
        next_idx = jr.choice(subkey, NUM_PAIRS, p=distribution)
        next_i = int(i_all[next_idx].astype(int))
        next_j = int(j_all[next_idx].astype(int))
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
