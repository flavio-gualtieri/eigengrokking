# tests/test_data.py

from __future__ import annotations

import pytest

from eigengrok.data.modular import get_modular_addition_data
from eigengrok.data.vision import get_torchvision_data


def test_unknown_torchvision_dataset_raises_value_error(tmp_path):
    # The name is checked before anything is downloaded.
    with pytest.raises(ValueError, match="CIFAR10"):
        get_torchvision_data(dataset="CIFAR10", download_directory=str(tmp_path))


def test_modular_addition_split_and_labels():
    p = 11
    train, test, _ = get_modular_addition_data(modulus=p, train_frac=0.3, seed=0)
    assert len(train) + len(test) == p * p
    assert len(train) == round(0.3 * p * p)

    x, y = train.tensors
    assert (x[:, 2] == p).all()  # the "=" token
    assert ((x[:, 0] + x[:, 1]) % p == y).all()

    # No (a, b) pair appears in both splits.
    pairs = lambda ds: {tuple(r) for r in ds.tensors[0][:, :2].tolist()}
    assert not pairs(train) & pairs(test)
