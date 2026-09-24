import time

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, lax
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)

FIELD_DTYPES = {
    "alpha_t": jnp.float32,
    "frequency": jnp.int32,
    "key": jnp.uint32,
}


def _logsumexp_excluding_each(logits: jnp.ndarray) -> jnp.ndarray:
    """Return log(sum(exp(logits[j]))) for every exclusion j != i."""
    prefix = lax.associative_scan(jnp.logaddexp, logits)
    suffix = lax.associative_scan(jnp.logaddexp, logits, reverse=True)
    negative_infinity = jnp.full((1,), -jnp.inf, dtype=logits.dtype)
    before = jnp.concatenate((negative_infinity, prefix[:-1]))
    after = jnp.concatenate((suffix[1:], negative_infinity))
    return jnp.logaddexp(before, after)


def _pair_sampling_logits(
    frequency: jnp.ndarray, temperature: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    entity_logits = -frequency.astype(jnp.float32) / temperature
    # A pair's weight factorizes as exp(logit[i]) * exp(logit[j]). The first
    # draw includes the combined weight of every valid partner; the second is
    # then sampled conditionally with the first entity excluded.
    first_logits = entity_logits + _logsumexp_excluding_each(entity_logits)
    return entity_logits, first_logits


@jit
def _draw_next_pair(
    frequency: jnp.ndarray, key: jnp.ndarray, temperature: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Draw the same distribution as enumerating every unordered pair."""
    entity_logits, first_logits = _pair_sampling_logits(frequency, temperature)
    next_key, first_key, second_key = jr.split(key, 3)

    first = jr.categorical(first_key, first_logits)
    second_logits = entity_logits.at[first].set(-jnp.inf)
    second = jr.categorical(second_key, second_logits)

    left = jnp.minimum(first, second)
    right = jnp.maximum(first, second)
    frequency = frequency.at[left].add(1).at[right].add(1)
    return frequency, next_key, left, right


class BayesianDecisionProcess(BaseModel):
    K: int
    alpha_t: jnp.ndarray
    frequency: jnp.ndarray
    key: jnp.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def initialize_missing(cls, values: dict[str, object]) -> dict[str, object]:
        K = values.get("K", 0)

        defaults = {
            "alpha_t": lambda: jnp.ones(K, dtype=jnp.float32),
            "frequency": lambda: jnp.zeros(K, dtype=jnp.int32),
            "key": lambda: jr.PRNGKey(int(time.time_ns())),
        }

        for field, default_fn in defaults.items():
            if values.get(field) is None:
                values[field] = default_fn()

        return values

    @classmethod
    def create(cls, k: int) -> BayesianDecisionProcess:
        return cls(
            K=k,
            alpha_t=jnp.ones(k, dtype=jnp.float32),
            frequency=jnp.zeros(k, dtype=jnp.int32),
            key=jr.PRNGKey(int(time.time_ns())),
        )

    @field_validator(*FIELD_DTYPES.keys(), mode="before")
    @classmethod
    def ensure_correct_dtype(cls, v: object, info: ValidationInfo) -> jnp.ndarray:
        if info.field_name is None:
            raise ValueError("Validator is missing a field name")
        dtype = FIELD_DTYPES[info.field_name]
        return jnp.array(v, dtype=dtype)

    def get_alphas(self) -> np.ndarray:
        return np.array(self.alpha_t)

    def snapshot(self) -> dict[str, object]:
        return {
            "K": self.K,
            "alpha_t": self.alpha_t.tolist(),
            "frequency": self.frequency.tolist(),
            "key": self.key.tolist(),
        }

    def submit_comparison(self, i: int, j: int, winner: int):
        Y_ij = 1 if winner == i else -1
        self.alpha_t = BayesianDecisionProcess.MM(self.alpha_t, i, j, Y_ij)

    def get_next_pair(self, temp: float = 1.0) -> tuple[int, int]:
        if self.K < 2:
            raise ValueError("At least two entities are required to draw a pair")
        if temp <= 0:
            raise ValueError("Pair sampling temperature must be positive")
        self.frequency, self.key, next_i, next_j = _draw_next_pair(
            self.frequency, self.key, temp
        )
        return int(next_i), int(next_j)

    @staticmethod
    @jit
    def MM(alpha_t: jnp.ndarray, i: int, j: int, Y_ij: int) -> jnp.ndarray:
        alpha_0 = jnp.sum(alpha_t)

        c = alpha_t / alpha_0
        c_ij_denom = alpha_0 * (alpha_t[i] + alpha_t[j] + 1.0)
        c = c.at[i].set(
            ((alpha_t[i] + (1.0 + Y_ij) / 2.0) * (alpha_t[i] + alpha_t[j])) / c_ij_denom
        )
        c = c.at[j].set(
            ((alpha_t[j] + (1.0 - Y_ij) / 2.0) * (alpha_t[i] + alpha_t[j])) / c_ij_denom
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

        sum_ck_sq = jnp.sum(c**2)
        alpha_0_prime = (D - 1.0) / (sum_ck_sq - D)
        alpha_prime = c * alpha_0_prime

        return alpha_prime

    @staticmethod
    @jit
    def softmax(logits: jnp.ndarray, temp: float = 1.0) -> jnp.ndarray:
        scaled = logits / temp
        exped = jnp.exp(scaled - jnp.max(scaled, axis=-1, keepdims=True))
        normed = exped / jnp.sum(exped, axis=-1, keepdims=True)
        return normed
