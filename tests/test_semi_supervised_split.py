"""Unit tests for the independent semi-supervised 80/20 split."""

import numpy as np

from core.run_semi_supervised import split_semi_supervised_data


def test_semi_supervised_split_is_80_20_and_disjoint():
    normal = np.arange(10, dtype=np.int32)[:, None]
    attack = np.arange(15, dtype=np.int32)[:, None]

    normal_train, normal_test, attack_train, attack_test = (
        split_semi_supervised_data(normal, attack, seed=7)
    )

    assert len(normal_train) == 8
    assert len(normal_test) == 2
    assert len(attack_train) == 12
    assert len(attack_test) == 3
    assert not set(normal_train.ravel()) & set(normal_test.ravel())
    assert not set(attack_train.ravel()) & set(attack_test.ravel())


def test_semi_supervised_split_is_reproducible():
    normal = np.arange(20, dtype=np.int32)[:, None]
    attack = np.arange(20, dtype=np.int32)[:, None]

    first = split_semi_supervised_data(normal, attack, seed=42)
    second = split_semi_supervised_data(normal, attack, seed=42)

    for first_part, second_part in zip(first, second):
        assert np.array_equal(first_part, second_part)
