import time
from typing import Self

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

from app.settings import settings

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


def _entity_logits(
    frequency: jnp.ndarray, temperature: float, active: jnp.ndarray
) -> jnp.ndarray:
    # Shifting by the minimum keeps the distribution and keeps float32 logits
    # near zero as frequencies grow. Inactive entities are removed before the
    # log-sum-exp so they contribute no weight and cannot be drawn.
    shifted = frequency - jnp.min(frequency)
    logits = -shifted.astype(jnp.float32) / temperature
    return jnp.where(active, logits, jnp.asarray(-jnp.inf, dtype=logits.dtype))


def _pair_sampling_logits(
    frequency: jnp.ndarray, temperature: float, active: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    entity_logits = _entity_logits(frequency, temperature, active)
    # A pair's weight factorizes as exp(logit[i]) * exp(logit[j]). The first
    # draw includes the combined weight of every valid partner; the second is
    # then sampled conditionally with the first entity excluded.
    first_logits = entity_logits + _logsumexp_excluding_each(entity_logits)
    return entity_logits, first_logits


def _commit_pair(
    frequency: jnp.ndarray,
    key: jnp.ndarray,
    i: jnp.ndarray,
    j: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    left = jnp.minimum(i, j)
    right = jnp.maximum(i, j)
    return frequency.at[left].add(1).at[right].add(1), key, left, right


@jit
def _draw_active_pair(
    frequency: jnp.ndarray,
    key: jnp.ndarray,
    temperature: float,
    active: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Draw the same distribution as enumerating every unordered active pair."""
    entity_logits, first_logits = _pair_sampling_logits(
        frequency, temperature, active
    )
    next_key, first_key, second_key = jr.split(key, 3)

    first = jr.categorical(first_key, first_logits)
    second_logits = entity_logits.at[first].set(-jnp.inf)
    second = jr.categorical(second_key, second_logits)
    return _commit_pair(frequency, next_key, first, second)


@jit
def _draw_forced_pair(
    frequency: jnp.ndarray,
    key: jnp.ndarray,
    temperature: float,
    active: jnp.ndarray,
    forced: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pair one required project with a partner drawn by inverse frequency."""
    entity_logits = _entity_logits(frequency, temperature, active)
    next_key, partner_key = jr.split(key, 2)
    partner_logits = entity_logits.at[forced].set(-jnp.inf)
    partner = jr.categorical(partner_key, partner_logits)
    return _commit_pair(frequency, next_key, forced, partner)


def _draw_next_pair(
    frequency: jnp.ndarray,
    key: jnp.ndarray,
    temperature: float,
    active: jnp.ndarray,
    min_judgments: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fill each active project up to min_judgments, then sample as usual.

    A count of zero disables the floor. Removed projects stay out because they
    are already missing from ``active``.
    """
    active_mask = np.asarray(active, dtype=bool)
    under = active_mask & (np.asarray(frequency) < min_judgments)
    short = int(under.sum())
    if short == 1:
        forced = jnp.int32(int(np.flatnonzero(under)[0]))
        return _draw_forced_pair(
            frequency, key, temperature, jnp.asarray(active_mask), forced
        )
    # Two or more short: draw only among them. None short: draw among every
    # project still active.
    mask = under if short >= 2 else active_mask
    return _draw_active_pair(frequency, key, temperature, jnp.asarray(mask))


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

    @model_validator(mode="after")
    def ensure_matching_dimensions(self) -> Self:
        expected_shape = (self.K,)
        if (
            self.alpha_t.shape != expected_shape
            or self.frequency.shape != expected_shape
        ):
            raise ValueError("alpha_t and frequency must each contain K values")
        return self

    def get_alphas(self) -> np.ndarray:
        return np.array(self.alpha_t)

    def snapshot(self) -> dict[str, object]:
        return {
            "K": self.K,
            "alpha_t": self.alpha_t.tolist(),
            "frequency": self.frequency.tolist(),
            "key": self.key.tolist(),
        }

    def propose_comparison(self, i: int, j: int, winner: int) -> jnp.ndarray:
        outcome = 1 if winner == i else -1
        return BayesianDecisionProcess.MM(self.alpha_t, i, j, outcome)

    def propose_pair(
        self,
        active: np.ndarray,
        temp: float = 1.0,
        *,
        frequency: jnp.ndarray | None = None,
        key: jnp.ndarray | None = None,
        min_judgments: int | None = None,
    ) -> tuple[int, int, jnp.ndarray, jnp.ndarray]:
        if self.K < 2:
            raise ValueError("At least two entities are required to draw a pair")
        if not temp > 0:
            raise ValueError("Pair sampling temperature must be positive")
        active_array = np.asarray(active, dtype=bool)
        if active_array.shape != (self.K,):
            raise ValueError("Active mask must contain K values")
        if int(active_array.sum()) < 2:
            raise ValueError("At least two entities are required to draw a pair")
        if frequency is None:
            frequency = self.frequency
        if key is None:
            key = self.key
        if min_judgments is None:
            min_judgments = settings.MIN_JUDGMENTS
        frequency, key, next_i, next_j = _draw_next_pair(
            frequency,
            key,
            temp,
            jnp.asarray(active_array),
            min_judgments,
        )
        return int(next_i), int(next_j), frequency, key

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
