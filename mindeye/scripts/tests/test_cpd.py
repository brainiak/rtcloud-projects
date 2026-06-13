"""Tests for the CPD (cortical pairmate distinctiveness) analysis.

Pure-logic + mocked-orchestration only -- no GPU, no real CLIP, no fMRI data.
Run with the project venv:  python -m pytest scripts/tests/test_cpd.py -v
"""
import numpy as np
import pandas as pd
import pytest
import torch

from utils_mindeye import cpd_from_embeddings, compute_cpd
from cpd_analysis import (
    foil_path_for,
    causal_volume_indices,
    avg_bold_volume_indices,
    build_lss_events,
    zscore_betas,
    causal_zscore_betas,
    per_trial_cpd,
    pairmate_2afc,
    pairmate_2afc_accuracy,
    forward_retrieval_accuracy,
)


# --------------------------------------------------------------------------
# foil_path_for
# --------------------------------------------------------------------------
class TestFoilPathFor:
    AVAIL = [
        "pair_9_13_44.png",
        "pair_9_13_89.png",
        "pair_10_14_33.png",
        "pair_10_14_89.png",
    ]

    def test_basic_foil(self):
        foil = foil_path_for(
            "all_stimuli/MST_pairs_styleGAN/pair_9_13_44.png", available=self.AVAIL
        )
        assert foil == "all_stimuli/MST_pairs_styleGAN/pair_9_13_89.png"

    def test_symmetric(self):
        foil = foil_path_for(
            "all_stimuli/MST_pairs_styleGAN/pair_9_13_89.png", available=self.AVAIL
        )
        assert foil.endswith("pair_9_13_44.png")

    def test_multidigit_prefix(self):
        foil = foil_path_for(
            "x/pair_10_14_33.png", available=self.AVAIL
        )
        assert foil == "x/pair_10_14_89.png"

    def test_bare_basename_input(self):
        foil = foil_path_for("pair_9_13_44.png", available=self.AVAIL)
        assert foil == "pair_9_13_89.png"

    def test_no_pairmate_raises(self):
        with pytest.raises(ValueError):
            foil_path_for("pair_9_13_44.png", available=["pair_9_13_44.png"])

    def test_ambiguous_raises(self):
        bad = ["pair_9_13_44.png", "pair_9_13_89.png", "pair_9_13_11.png"]
        with pytest.raises(ValueError):
            foil_path_for("pair_9_13_44.png", available=bad)


# --------------------------------------------------------------------------
# cpd_from_embeddings  (the pure projection math)
# --------------------------------------------------------------------------
class TestCpdFromEmbeddings:
    def test_correct_is_plus_one(self):
        correct = torch.tensor([2.0, 0.0])
        foil = torch.tensor([0.0, 2.0])
        assert cpd_from_embeddings(correct, correct, foil, normalize=False) == pytest.approx(1.0)

    def test_foil_is_minus_one(self):
        correct = torch.tensor([2.0, 0.0])
        foil = torch.tensor([0.0, 2.0])
        assert cpd_from_embeddings(foil, correct, foil, normalize=False) == pytest.approx(-1.0)

    def test_midpoint_is_zero(self):
        correct = torch.tensor([2.0, 0.0])
        foil = torch.tensor([0.0, 2.0])
        mid = (correct + foil) / 2
        assert cpd_from_embeddings(mid, correct, foil, normalize=False) == pytest.approx(0.0)

    def test_beyond_correct_exceeds_one(self):
        # pred = [3, -1]; mid = [1,1]; axis = [2,-2]; cpd = 2*8/8 = 2
        correct = torch.tensor([2.0, 0.0])
        foil = torch.tensor([0.0, 2.0])
        pred = torch.tensor([3.0, -1.0])
        assert cpd_from_embeddings(pred, correct, foil, normalize=False) == pytest.approx(2.0)

    def test_perpendicular_component_ignored(self):
        # moving off the axis (along [1,1]) from the midpoint does not change CPD
        correct = torch.tensor([2.0, 0.0])
        foil = torch.tensor([0.0, 2.0])
        pred = torch.tensor([2.0, 2.0])  # midpoint [1,1] + [1,1]
        assert cpd_from_embeddings(pred, correct, foil, normalize=False) == pytest.approx(0.0)

    def test_normalize_makes_magnitude_irrelevant(self):
        # after L2 normalization, correct->[1,0], foil->[0,1], pred(scaled)->[1,0] => +1
        correct = torch.tensor([3.0, 0.0])
        foil = torch.tensor([0.0, 5.0])
        pred = torch.tensor([10.0, 0.0])
        assert cpd_from_embeddings(pred, correct, foil, normalize=True) == pytest.approx(1.0)

    def test_accepts_multidim_and_flattens(self):
        correct = torch.zeros(1, 2, 2)
        correct[0, 0, 0] = 2.0
        foil = torch.zeros(1, 2, 2)
        foil[0, 0, 1] = 2.0
        # pred == correct -> +1 regardless of the (1,2,2) shape
        assert cpd_from_embeddings(correct, correct, foil, normalize=False) == pytest.approx(1.0)

    def test_numpy_inputs_supported(self):
        correct = np.array([2.0, 0.0])
        foil = np.array([0.0, 2.0])
        assert cpd_from_embeddings(correct, correct, foil, normalize=False) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# compute_cpd  (image-embedder wiring around cpd_from_embeddings)
# --------------------------------------------------------------------------
class _FakeEmbedder:
    """Returns preset embeddings in call order; records the inputs it saw."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.seen = []

    def __call__(self, x):
        self.seen.append(x)
        return self._outputs.pop(0)


class TestComputeCpd:
    def _imgs(self, imsize=2):
        return torch.zeros(3, imsize, imsize), torch.ones(3, imsize, imsize)

    def test_matches_pure_helper_correct(self):
        A = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        B = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        emb = _FakeEmbedder([A, B])
        correct_img, foil_img = self._imgs()
        clipvoxel = A.clone()  # predicted == correct => +1
        out = compute_cpd(emb, clipvoxel, correct_img, foil_img, imsize=2, device="cpu")
        assert out == pytest.approx(1.0)

    def test_matches_pure_helper_foil(self):
        A = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        B = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        emb = _FakeEmbedder([A, B])
        correct_img, foil_img = self._imgs()
        clipvoxel = B.clone()
        out = compute_cpd(emb, clipvoxel, correct_img, foil_img, imsize=2, device="cpu")
        assert out == pytest.approx(-1.0)

    def test_embeds_correct_then_foil(self):
        A = torch.tensor([[1.0, 0.0]])
        B = torch.tensor([[0.0, 1.0]])
        emb = _FakeEmbedder([A, B])
        correct_img, foil_img = self._imgs()
        compute_cpd(emb, A.clone(), correct_img, foil_img, imsize=2, device="cpu")
        # first call embeds the correct image, second the foil
        assert torch.equal(emb.seen[0].reshape(-1), correct_img.reshape(-1))
        assert torch.equal(emb.seen[1].reshape(-1), foil_img.reshape(-1))

    def test_consistent_with_cpd_from_embeddings(self):
        A = torch.tensor([[0.3, 0.7, 0.1]])
        B = torch.tensor([[0.9, 0.0, 0.2]])
        pred = torch.tensor([[0.5, 0.4, 0.6]])
        emb = _FakeEmbedder([A, B])
        correct_img, foil_img = self._imgs()
        out = compute_cpd(emb, pred.clone(), correct_img, foil_img, imsize=2, device="cpu")
        expected = cpd_from_embeddings(pred, A, B, normalize=True)
        assert out == pytest.approx(float(expected))


# --------------------------------------------------------------------------
# causal_volume_indices
# --------------------------------------------------------------------------
class TestCausalVolumeIndices:
    def test_basic(self):
        # onset 0, L=3, tail=8 => cutoff 11; floor(11/1.5)=7 -> 0..7
        idx = causal_volume_indices(0.0, 3.0, tr_length=1.5, n_vols=239, hrf_tail=8.0)
        assert idx == list(range(0, 8))

    def test_starts_at_run_start(self):
        idx = causal_volume_indices(30.0, 21.0, tr_length=1.5, n_vols=239, hrf_tail=8.0)
        assert idx[0] == 0
        # cutoff 59 -> floor(59/1.5)=39
        assert idx[-1] == 39

    def test_capped_at_n_vols(self):
        # cutoff far past the end of the run -> clamp to the last available volume
        idx = causal_volume_indices(1000.0, 21.0, tr_length=1.5, n_vols=239, hrf_tail=8.0)
        assert idx[-1] == 238
        assert len(idx) == 239

    def test_contiguous(self):
        idx = causal_volume_indices(15.0, 5.0, tr_length=1.5, n_vols=239, hrf_tail=8.0)
        assert idx == list(range(idx[0], idx[-1] + 1))


# --------------------------------------------------------------------------
# avg_bold_volume_indices
# --------------------------------------------------------------------------
class TestAvgBoldVolumeIndices:
    def test_onset_zero(self):
        # midpoints (i+0.5)*1.5 in [4,8] -> i=3 (5.25), i=4 (6.75)
        idx = avg_bold_volume_indices(0.0, tr_length=1.5, n_vols=239)
        assert idx == [3, 4]

    def test_onset_thirty(self):
        idx = avg_bold_volume_indices(30.0, tr_length=1.5, n_vols=239)
        assert idx == [23, 24]

    def test_custom_window(self):
        idx = avg_bold_volume_indices(0.0, tr_length=1.5, n_vols=239, lo=4.0, hi=10.0)
        # midpoints in [4,10]: i=3(5.25),4(6.75),5(8.25),6(9.75)
        assert idx == [3, 4, 5, 6]

    def test_respects_n_vols_bound(self):
        idx = avg_bold_volume_indices(0.0, tr_length=1.5, n_vols=4)
        assert all(0 <= i < 4 for i in idx)
        assert idx == [3]


# --------------------------------------------------------------------------
# build_lss_events
# --------------------------------------------------------------------------
class TestBuildLssEvents:
    def _events(self):
        return pd.DataFrame(
            {
                "onset": [100.0, 130.0, 160.0],
                "duration": [21.0, 21.0, 21.0],
                "trial_number": [0, 1, 2],
                "image_name": ["a.png", "b.png", "c.png"],
            }
        )

    def test_probe_and_reference_labels(self):
        out = build_lss_events(self._events(), probe_trial_number=1, probe_duration=3.0)
        assert set(out["trial_type"]) == {"probe", "reference"}
        probe = out[out["trial_type"] == "probe"]
        assert len(probe) == 1
        assert probe["duration"].iloc[0] == pytest.approx(3.0)

    def test_causal_drops_future_trials(self):
        out = build_lss_events(self._events(), probe_trial_number=1, probe_duration=3.0)
        # trials 0 and 1 only (trial 2 is in the future)
        assert len(out) == 2

    def test_onsets_rezeroed(self):
        out = build_lss_events(self._events(), probe_trial_number=1, probe_duration=3.0)
        assert out["onset"].min() == pytest.approx(0.0)
        probe = out[out["trial_type"] == "probe"]
        assert probe["onset"].iloc[0] == pytest.approx(30.0)

    def test_references_keep_true_duration(self):
        out = build_lss_events(self._events(), probe_trial_number=1, probe_duration=3.0)
        ref = out[out["trial_type"] == "reference"]
        assert ref["duration"].tolist() == pytest.approx([21.0] * len(ref))

    def test_required_columns_present(self):
        out = build_lss_events(self._events(), probe_trial_number=2, probe_duration=7.0)
        for col in ("onset", "duration", "trial_type"):
            assert col in out.columns
        # full session up to probe -> 3 trials
        assert len(out) == 3


# --------------------------------------------------------------------------
# Aggregation helpers (mocked predict / embeddings; no GPU/data)
# --------------------------------------------------------------------------
class TestZscoreBetas:
    def test_zero_mean_unit_std_per_voxel(self):
        b = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        z = zscore_betas(b)
        assert z.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)
        assert z.std(axis=0) == pytest.approx([1.0, 1.0], abs=1e-3)


class TestCausalZscoreBetas:
    def test_row0_is_zero(self):
        # first trial has only itself -> std=0 -> z=0 (degenerate, by design)
        b = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        z = causal_zscore_betas(b)
        assert z[0] == pytest.approx([0.0, 0.0], abs=1e-9)

    def test_each_row_uses_only_prefix_stats(self):
        b = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        z = causal_zscore_betas(b)
        # t=1: mean=[1.5,15] std=[0.5,5] -> ~[1,1]
        assert z[1] == pytest.approx([1.0, 1.0], abs=1e-3)
        # t=2: mean=[2,20] std=sqrt(2/3)*[1,10] -> [1.2247,1.2247]
        assert z[2] == pytest.approx([1.22474, 1.22474], abs=1e-3)

    def test_last_row_matches_global_but_earlier_rows_differ(self):
        b = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        g = zscore_betas(b)
        c = causal_zscore_betas(b)
        # final row sees the whole prefix == full session -> identical to global
        assert c[-1] == pytest.approx(g[-1], abs=1e-6)
        # but the first row differs (global uses future trials, causal does not)
        assert not np.allclose(c[0], g[0])


class TestPerTrialCpd:
    def test_predict_correct_gives_plus_one(self):
        n_vox = 4
        betas = np.random.RandomState(0).randn(3, n_vox)
        correct = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])]
        foil = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])]
        # predict_fn returns each trial's correct embedding -> CPD == +1
        state = {"t": 0}

        def predict_fn(_betas_tt):
            emb = correct[state["t"]]
            state["t"] += 1
            return emb

        cpd = per_trial_cpd(betas, correct, foil, predict_fn, num_voxels=n_vox)
        assert cpd == pytest.approx([1.0, 1.0, 1.0])

    def test_betas_tt_shape_passed_to_predict(self):
        n_vox = 5
        betas = np.zeros((2, n_vox))
        seen = []

        def predict_fn(betas_tt):
            seen.append(tuple(betas_tt.shape))
            return torch.tensor([1.0, 0.0])

        per_trial_cpd(betas, [torch.tensor([1.0, 0.0])] * 2,
                      [torch.tensor([0.0, 1.0])] * 2, predict_fn, num_voxels=n_vox)
        assert seen == [(1, 1, n_vox), (1, 1, n_vox)]


class TestRetrievalMetrics:
    def test_2afc_correct_closer(self):
        pred = torch.tensor([1.0, 0.0])
        correct = torch.tensor([1.0, 0.0])
        foil = torch.tensor([0.0, 1.0])
        assert pairmate_2afc(pred, correct, foil) == 1.0
        assert pairmate_2afc(foil, correct, foil) == 0.0

    def test_2afc_accuracy_mean(self):
        preds = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        correct = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])]
        foil = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])]
        # trial 0 correct closer (1.0), trial 1 foil closer (0.0) -> 0.5
        assert pairmate_2afc_accuracy(preds, correct, foil) == pytest.approx(0.5)

    def test_forward_retrieval_perfect(self):
        pool = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        preds = [torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9])]
        assert forward_retrieval_accuracy(preds, [0, 1], pool) == pytest.approx(1.0)

    def test_forward_retrieval_wrong(self):
        pool = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        preds = [torch.tensor([0.1, 0.9])]  # closest to pool[1], but correct is 0
        assert forward_retrieval_accuracy(preds, [0], pool) == pytest.approx(0.0)
