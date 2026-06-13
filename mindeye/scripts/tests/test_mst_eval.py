"""Tests for the minimally-modified pairmate 2-AFC eval (mst_eval.py).

No GPU / no real CLIP / no fMRI: a fake one-hot decoder + embedder make the
2-AFC outcome exactly predictable, so we can lock in both the scoring logic and
the index-space contract (single-trial AND averaged), plus every guard rail.

Run:  python -m pytest scripts/tests/test_mst_eval.py -v
"""
import numpy as np
import pytest
import torch

from mst_eval import (
    evaluate_mst_pairs,
    assert_pairmate_inputs,
    batchwise_cosine_similarity,
)

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Fake one-hot world: each image/voxel row carries an integer morph id.
#   - embedder(image) -> one-hot(morph_id)
#   - model decodes voxel -> one-hot(decode_id)
# So cosine(decode, target) == 1 iff decode_id == target's morph id, else 0.
# A "perfect" decoder sets each voxel row's decode_id to its own image's id.
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    def __init__(self, n_classes):
        self.n = n_classes

    def __call__(self, image):                       # image: (1, ...) holding an id
        ids = image.reshape(image.shape[0], -1)[:, 0].long()
        oh = torch.zeros(ids.shape[0], self.n)
        oh[torch.arange(ids.shape[0]), ids] = 1.0
        return oh                                    # (1, n)


class FakeModel:
    def __init__(self, n_classes):
        self.n = n_classes

    def ridge(self, voxel, _):                       # passthrough
        return voxel

    def backbone(self, voxel_ridge):                 # voxel holds the decode id
        ids = voxel_ridge.reshape(-1).long()
        oh = torch.zeros(ids.shape[0], self.n)
        oh[torch.arange(ids.shape[0]), ids] = 1.0
        return None, oh, None                        # (backbone, clip_voxels, blurry)


def make_world(n_pairs=4):
    """Build a synthetic MST dataset with `n_pairs` pairs (2 morphs each),
    each unique morph presented twice (two presentation blocks)."""
    pairs_meta, morphs = [], []
    for k in range(n_pairs):
        pre = f"pair_{k}_{k + 1}"
        m0, m1 = f"{pre}_0.png", f"{pre}_1.png"
        morphs += [m0, m1]
        pairs_meta.append((m0, m1))
    n_unique = len(morphs)                           # 2 * n_pairs
    block1 = list(morphs)                            # ordering A
    block2 = list(reversed(morphs))                  # ordering B (non-trivial)
    pres_names = np.array(block1 + block2)           # length 2 * n_unique
    n_pres = len(pres_names)

    # pairs as PRESENTATION indices, exactly like the offline x.index(...) halves
    half = n_unique
    x1, x2 = list(pres_names[:half]), list(pres_names[half:])
    rows = []
    for m0, m1 in pairs_meta:
        rows.append((x1.index(m0), x1.index(m1)))
    for m0, m1 in pairs_meta:
        rows.append((x2.index(m0) + half, x2.index(m1) + half))
    pairs = np.array(rows)

    unique_images = np.unique(pres_names)            # sorted, length n_unique
    name2id = {n: i for i, n in enumerate(unique_images)}
    pres2uid = np.array([name2id[n] for n in pres_names])

    return dict(pres_names=pres_names, unique_images=unique_images,
                name2id=name2id, pres2uid=pres2uid, pairs=pairs,
                n_unique=n_unique, n_pres=n_pres)


def _rows_tensor(ids):
    """One id per row -> (R, 1) float tensor (so [None] -> (1,1) for the fakes)."""
    return torch.tensor(np.asarray(ids, dtype=np.float32)).reshape(-1, 1)


# ---- single-trial inputs (per-presentation vox/images, identity image_idx) -- #
def single_trial_inputs(w, decode_offset=None):
    ids = np.array([w["name2id"][n] for n in w["pres_names"]])
    images = _rows_tensor(ids)
    decode = ids.copy()
    if decode_offset is not None:
        for row, off in decode_offset.items():       # corrupt specific rows
            decode[row] = (decode[row] + off) % w["n_unique"]
    vox = _rows_tensor(decode)
    return dict(vox=vox, images=images,
                image_idx=np.arange(w["n_pres"]),
                vox_image_names=w["pres_names"])


# ---- averaged inputs (per-unique vox/images, presentation->uid image_idx) ---- #
def averaged_inputs(w):
    ids = np.arange(w["n_unique"])                    # unique_images sorted -> id == row
    images = _rows_tensor(ids)
    vox = _rows_tensor(ids)                           # perfect decoder
    return dict(vox=vox, images=images,
                image_idx=w["pres2uid"],
                vox_image_names=w["unique_images"])


def run(w, inp, **kw):
    return evaluate_mst_pairs(
        w["pairs"], vox=inp["vox"], images=inp["images"],
        image_idx=inp["image_idx"], model=FakeModel(w["n_unique"]),
        clip_img_embedder=FakeEmbedder(w["n_unique"]), device=DEVICE,
        data_type=torch.float32, vox_image_names=inp["vox_image_names"], **kw)


# =========================================================================== #
# scoring logic + both index-space modes
# =========================================================================== #
def test_perfect_decoder_single_trial_scores_1():
    w = make_world()
    assert run(w, single_trial_inputs(w)) == 1.0


def test_perfect_decoder_averaged_scores_1():
    w = make_world()
    assert run(w, averaged_inputs(w)) == 1.0


def test_anti_decoder_scores_0():
    # every voxel decodes to its pairmate's id -> always picks the foil
    w = make_world()
    inp = single_trial_inputs(w)
    ids = np.array([w["name2id"][n] for n in w["pres_names"]])
    # pairmate id: even<->odd morph within a pair (ids are [m0,m1] per pair, contiguous after sort?)
    # build foil id from names instead of assuming layout:
    foil_id = {}
    for a, b in w["pairs"]:
        ia = ids[a]; ib = ids[b]
        foil_id[ia] = ib; foil_id[ib] = ia
    inp["vox"] = _rows_tensor([foil_id[i] for i in ids])
    assert run(w, inp) == 0.0


def test_one_corrupted_voxel_drops_score_by_one_comparison():
    # corrupt a single presentation's decode -> exactly one of n_pres comparisons flips
    w = make_world()
    inp = single_trial_inputs(w, decode_offset={3: 1})
    expected = (w["n_pres"] - 1) / w["n_pres"]
    assert run(w, inp) == pytest.approx(expected)


def test_single_trial_and_averaged_agree():
    w = make_world()
    assert run(w, single_trial_inputs(w)) == run(w, averaged_inputs(w))


# =========================================================================== #
# the actual bug: presentation-ordered vox with a unique-id image_idx
# =========================================================================== #
def test_buggy_combo_is_rejected():
    w = make_world()
    inp = single_trial_inputs(w)          # vox is per-presentation (len n_pres)
    inp["image_idx"] = w["pres2uid"]      # but image_idx is the unique-id map  <-- the bug
    with pytest.raises(AssertionError, match="cover every vox row"):
        run(w, inp)


def test_buggy_combo_is_invisible_to_a_perfect_decoder():
    # subtle: voxel & target share the same index, so a *perfect* decoder still
    # scores 1.0 under the buggy image_idx -- the score does NOT reveal the bug.
    # This is precisely why the assertion guard (not the number) is the protection.
    w = make_world()
    inp = single_trial_inputs(w)
    inp["image_idx"] = w["pres2uid"]
    assert run(w, inp, check_inputs=False) == 1.0


def test_buggy_combo_gives_wrong_score_with_imperfect_decoder():
    # imperfect decoder: block-1 voxels perfect, block-2 voxels corrupted.
    # correct (identity) evaluates all 16 -> 8 right -> 0.5.
    # buggy (unique-id image_idx) only ever reads the perfect block-1 rows
    # -> a misleadingly inflated number != the true score. The guard would catch it.
    w = make_world()
    corrupt = {r: 1 for r in range(w["n_unique"], w["n_pres"])}   # block-2 rows wrong
    inp = single_trial_inputs(w, decode_offset=corrupt)
    correct = run(w, inp)                                          # identity image_idx
    assert correct == pytest.approx(0.5)

    inp["image_idx"] = w["pres2uid"]
    buggy = run(w, inp, check_inputs=False)
    assert buggy != correct                                       # bug changed the result


# =========================================================================== #
# guard-rail unit tests (assert_pairmate_inputs)
# =========================================================================== #
def _args(w, inp):
    return dict(mst_pairs=w["pairs"], image_idx=inp["image_idx"],
                vox=inp["vox"], images=inp["images"],
                vox_image_names=inp["vox_image_names"])


def test_guard_passes_on_valid_single_trial():
    w = make_world()
    assert_pairmate_inputs(**_args(w, single_trial_inputs(w)))   # no raise


def test_guard_passes_on_valid_averaged():
    w = make_world()
    assert_pairmate_inputs(**_args(w, averaged_inputs(w)))       # no raise


def test_guard_rejects_out_of_range_pair():
    w = make_world()
    a = _args(w, single_trial_inputs(w))
    a["mst_pairs"] = w["pairs"].copy()
    a["mst_pairs"][0, 0] = w["n_pres"]               # out of image_idx range
    with pytest.raises(AssertionError, match="index image_idx"):
        assert_pairmate_inputs(**a)


def test_guard_rejects_self_pair():
    w = make_world()
    a = _args(w, single_trial_inputs(w))
    a["mst_pairs"] = w["pairs"].copy()
    a["mst_pairs"][0, 1] = a["mst_pairs"][0, 0]      # target == foil index
    with pytest.raises(AssertionError, match="self-paired"):
        assert_pairmate_inputs(**a)


def test_guard_rejects_dead_rows():
    # drop a presentation from coverage by duplicating another -> a vox row never hit
    w = make_world()
    a = _args(w, single_trial_inputs(w))
    a["mst_pairs"] = w["pairs"].copy()
    a["mst_pairs"][0, 0] = a["mst_pairs"][1, 0]      # now some row reached twice, another never
    with pytest.raises(AssertionError, match="cover every vox row|non-uniform"):
        assert_pairmate_inputs(**a)


def test_guard_rejects_non_pairmate_foil():
    w = make_world()
    a = _args(w, single_trial_inputs(w))
    names = np.array(a["vox_image_names"], dtype=object)
    names[0] = "pair_999_999_0.png"                  # break the pairmate relationship
    a["vox_image_names"] = names
    with pytest.raises(AssertionError, match="not pairmates|same morph"):
        assert_pairmate_inputs(**a)


def test_guard_rejects_misaligned_vox_images():
    w = make_world()
    a = _args(w, single_trial_inputs(w))
    a["images"] = a["images"][:-1]                   # length mismatch
    with pytest.raises(AssertionError, match="must be aligned"):
        assert_pairmate_inputs(**a)


# =========================================================================== #
# sanity on the cosine helper
# =========================================================================== #
def test_cosine_helper_matches_dot_for_unit_vectors():
    a = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    assert batchwise_cosine_similarity(a, b).item() == pytest.approx(
        (a @ b.T).item(), abs=1e-5)
