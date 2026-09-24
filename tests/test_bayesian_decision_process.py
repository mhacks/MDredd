import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from app.algorithm.bayesian_decision_process import (
    BayesianDecisionProcess,
    _pair_sampling_logits,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


@pytest.mark.parametrize(
    ("frequency", "temperature"),
    [
        ([0, 0], 1.0),
        ([0, 0, 0, 0], 1.0),
        ([0, 1, 2, 7], 1.0),
        ([3, 1, 8, 2, 5], 0.5),
        ([0, 1_000, 1_001, 1_002], 1.0),
    ],
)
def test_factorized_distribution_matches_enumerated_pairs(
    frequency: list[int], temperature: float
) -> None:
    counts = np.asarray(frequency, dtype=np.int32)
    left, right = np.triu_indices(len(counts), k=1)
    expected = _softmax(-(counts[left] + counts[right]) / temperature)

    entity_logits, first_logits = _pair_sampling_logits(
        jnp.asarray(counts), temperature
    )
    entity_logits = np.asarray(entity_logits)
    first_probabilities = _softmax(np.asarray(first_logits))

    actual: list[float] = []
    for i, j in zip(left, right, strict=True):
        given_i = entity_logits.copy()
        given_i[i] = -np.inf
        given_j = entity_logits.copy()
        given_j[j] = -np.inf
        probability = (
            first_probabilities[i] * _softmax(given_i)[j]
            + first_probabilities[j] * _softmax(given_j)[i]
        )
        actual.append(float(probability))

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


def test_pair_draw_updates_only_two_distinct_entities() -> None:
    model = BayesianDecisionProcess(
        K=6,
        alpha_t=jnp.ones(6),
        frequency=jnp.asarray([0, 1, 2, 3, 4, 5]),
        key=jr.PRNGKey(42),
    )

    for _ in range(25):
        before = np.asarray(model.frequency)
        left, right = model.get_next_pair()
        delta = np.asarray(model.frequency) - before

        assert 0 <= left < right < model.K
        assert delta[left] == 1
        assert delta[right] == 1
        assert np.count_nonzero(delta) == 2


def test_pair_draw_is_deterministic_after_snapshot_round_trip() -> None:
    original = BayesianDecisionProcess(
        K=5,
        alpha_t=jnp.ones(5),
        frequency=jnp.asarray([4, 2, 0, 1, 3]),
        key=jr.PRNGKey(7),
    )
    restored = BayesianDecisionProcess.model_validate(original.snapshot())

    for _ in range(10):
        assert original.get_next_pair() == restored.get_next_pair()
        np.testing.assert_array_equal(original.frequency, restored.frequency)
        np.testing.assert_array_equal(original.key, restored.key)


@pytest.mark.parametrize("entity_count", [0, 1])
def test_pair_draw_requires_two_entities(entity_count: int) -> None:
    model = BayesianDecisionProcess.create(entity_count)

    with pytest.raises(ValueError, match="At least two entities"):
        model.get_next_pair()


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_pair_draw_requires_positive_temperature(temperature: float) -> None:
    model = BayesianDecisionProcess.create(2)

    with pytest.raises(ValueError, match="temperature must be positive"):
        model.get_next_pair(temperature)
