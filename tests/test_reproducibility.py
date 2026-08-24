import pytest

from qwen_math_challenge.reproducibility import deterministic_probe, seed_everything


def test_seed_reproduces_python_and_numpy_probe() -> None:
    first_report = seed_everything(20260824)
    first_probe = deterministic_probe()
    second_report = seed_everything(20260824)
    second_probe = deterministic_probe()

    assert first_probe == second_probe
    assert first_report["python_random_seeded"] is True
    assert second_report["numpy_seeded"] is True


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_rejects_invalid_seed(seed: int) -> None:
    with pytest.raises(ValueError):
        seed_everything(seed)
