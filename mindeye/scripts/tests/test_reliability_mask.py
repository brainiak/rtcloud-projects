import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "reliability_mask.py"
SPEC = importlib.util.spec_from_file_location("reliability_mask", MODULE_PATH)
reliability_mask = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reliability_mask)  # type: ignore[attr-defined]


def test_build_repeat_groups_excludes_test_images() -> None:
    names = np.array([
        "img_a.png",
        "img_a.png",
        "img_b.png",
        "img_b.png",
        "MST_pairs/img_c.png",
    ])
    groups = reliability_mask.build_repeat_groups(names, ["MST_pairs"])
    assert set(groups.keys()) == {"img_a.png", "img_b.png"}
    assert groups["img_a.png"] == [0, 1]
    assert groups["img_b.png"] == [2, 3]


def test_compute_reliability_two_repeats() -> None:
    betas = np.array([
        [1.0, 0.0],  # img a repeat 1
        [2.0, 0.0],  # img a repeat 2
        [3.0, 0.0],  # img b repeat 1
        [4.0, 0.0],  # img b repeat 2
    ])
    groups = {"img_a": [0, 1], "img_b": [2, 3]}
    reliability = reliability_mask.compute_reliability(betas, groups)
    assert reliability.shape == (2,)
    np.testing.assert_allclose(reliability[0], 1.0)
    assert reliability[1] == 0.0


def test_compute_reliability_three_repeats_uses_all_pairs() -> None:
    betas = np.array([
        [1.0, 0.0],  # img a repeat 1
        [2.0, 0.0],  # img a repeat 2
        [3.0, 0.0],  # img a repeat 3
        [4.0, 0.0],  # img b repeat 1
        [5.0, 0.0],  # img b repeat 2
        [6.0, 0.0],  # img b repeat 3
    ])
    groups = {"img_a": [0, 1, 2], "img_b": [3, 4, 5]}
    reliability = reliability_mask.compute_reliability(betas, groups)
    np.testing.assert_allclose(reliability[0], 1.0)
    assert reliability[1] == 0.0


def test_threshold_reliability_quantile() -> None:
    reliability = np.array([0.1, 0.2, 0.3, 0.4])
    mask, thresh = reliability_mask.threshold_reliability(reliability, None, 0.75)
    assert np.array_equal(mask, np.array([False, False, True, True]))
    assert pytest.approx(thresh, rel=1e-6) == 0.325


def test_format_session_suffix() -> None:
    assert reliability_mask.format_session_suffix(["ses-01", "ses-02"]) == "ses-01-02"
    assert reliability_mask.format_session_suffix(["custom", "ses-02"]) == "custom-ses-02"
    assert reliability_mask.format_session_suffix([]) == "sessions"


def test_load_betas_accepts_transposed(tmp_path: Path) -> None:
    trials_first = np.arange(12, dtype=np.float32).reshape(4, 3)
    path_first = tmp_path / "betas_trials_first.npy"
    np.save(path_first, trials_first)
    loaded_first = reliability_mask.load_betas(path_first, expected_trials=4)
    assert loaded_first.shape == (4, 3)
    np.testing.assert_allclose(loaded_first, trials_first)

    voxels_first = trials_first.T
    path_voxels = tmp_path / "betas_voxels_first.npy"
    np.save(path_voxels, voxels_first)
    loaded_voxels = reliability_mask.load_betas(path_voxels, expected_trials=4)
    assert loaded_voxels.shape == (4, 3)
    np.testing.assert_allclose(loaded_voxels, trials_first)

    path_bad = tmp_path / "betas_bad.npy"
    np.save(path_bad, np.ones((2, 2)))
    with pytest.raises(ValueError):
        reliability_mask.load_betas(path_bad, expected_trials=4)
