"""Generate voxel reliability maps and union masks for MindEye training sessions.

This script consumes GLMsingle beta estimates alongside design metadata to
calculate voxelwise reliability based on repeated presentations of training
images. Test images (e.g., MST pairs) are excluded from the calculation, and all
available repeats per image are used when estimating correlations.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    nib = None  # lazily checked when nifti output is requested

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import utils_mindeye  # noqa: E402


@dataclass(frozen=True)
class SessionReliability:
    session: str
    reliability: np.ndarray
    mask: np.ndarray
    threshold: float | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute voxel reliability masks for MindEye training sessions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("subject", help="Subject identifier, e.g. sub-005")
    parser.add_argument(
        "sessions",
        nargs="+",
        help="Training sessions to include, e.g. ses-01 ses-02",
    )
    parser.add_argument(
        "--task",
        default="C",
        help="Functional task label used in file naming",
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "conf" / "config.json"),
        help="Path to MindEye JSON config containing data/derivatives paths",
    )
    parser.add_argument(
        "--beta-pattern",
        default="{subject}_{session}_task-{task}_betas.npy",
        help=(
            "Filename template (relative to derivatives path) for GLMsingle betas. "
            "Use placeholders {subject}, {session}, {task}."
        ),
    )
    parser.add_argument(
        "--design-task",
        default=None,
        help="Override design task name if it differs from --task",
    )
    parser.add_argument(
        "--test-pattern",
        action="append",
        default=["MST_pairs"],
        help=(
            "Substring identifying held-out test images. May be provided multiple "
            "times; any match will be excluded from reliability calculations."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute reliability threshold (0-1).",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="Quantile (0-1) used to select top voxels per session if threshold is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs; defaults to data_path from config",
    )
    parser.add_argument(
        "--mask-img",
        default=None,
        help="Optional NIfTI file corresponding to the flattened voxel space."
             " When provided, a 3D union mask will be written alongside the" " 1D numpy array.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip sessions with missing betas instead of raising an error.",
    )
    parser.add_argument(
        "--name-prefix",
        default="union_mask_from",
        help="Prefix for the saved union mask filename.",
    )
    args = parser.parse_args(argv)

    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be between 0 and 1")
    if args.quantile is not None:
        if not (0.0 <= args.quantile <= 1.0):
            parser.error("--quantile must be between 0 and 1")
        if args.threshold is not None:
            parser.error("Specify at most one of --threshold or --quantile")
    return args


def load_config(config_path: Path) -> Mapping[str, str]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    required_keys = {"data_path", "derivatives_path"}
    missing = required_keys.difference(config)
    if missing:
        raise KeyError(f"Config missing required keys: {sorted(missing)}")
    return config


def resolve_output_dir(args: argparse.Namespace, config: Mapping[str, str]) -> Path:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["data_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_image_names(
    subject: str,
    session: str,
    task: str,
    design_dir: Path,
) -> np.ndarray:
    _, _, _, _, image_names, _, _ = utils_mindeye.load_design_files(
        sub=subject,
        session=session,
        func_task_name=task,
        designdir=str(design_dir),
    )
    return np.asarray(image_names).astype(str)


def find_beta_file(
    derivatives_path: Path,
    subject: str,
    session: str,
    task: str,
    template: str,
) -> Path:
    candidate = derivatives_path / template.format(subject=subject, session=session, task=task)
    if candidate.is_file():
        return candidate
    pattern = f"**/{candidate.name}"
    matches = list(derivatives_path.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not locate betas using template {template!r} under {derivatives_path}"\
            f" for session {session}."
        )
    raise FileExistsError(
        f"Multiple betas files matched template {template!r} under {derivatives_path}:"\
        f" {matches}"
    )


def load_betas(beta_path: Path, expected_trials: int) -> np.ndarray:
    betas = np.asarray(np.load(beta_path))
    if betas.ndim == 0:
        raise ValueError(f"Betas file {beta_path} was empty.")
    if betas.shape[0] == expected_trials:
        betas = betas.reshape(expected_trials, -1)
    elif betas.shape[-1] == expected_trials:
        betas = np.moveaxis(betas, -1, 0).reshape(expected_trials, -1)
    else:
        raise ValueError(
            f"Betas in {beta_path} have shape {betas.shape}, which does not align with "
            f"the expected number of trials ({expected_trials})."
        )
    return betas.astype(np.float32)


def build_repeat_groups(
    image_names: Sequence[str],
    test_patterns: Sequence[str],
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(image_names):
        if any(pat in name for pat in test_patterns):
            continue
        groups.setdefault(name, []).append(idx)
    groups = {k: sorted(v) for k, v in groups.items() if len(v) >= 2}
    if not groups:
        raise ValueError("No training images with at least two repeats were found.")
    return groups


def voxelwise_corr(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Correlation arrays must match in shape, got {a.shape} vs {b.shape}")
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    numerator = (a_centered * b_centered).sum(axis=0)
    denom = np.sqrt((a_centered ** 2).sum(axis=0) * (b_centered ** 2).sum(axis=0))
    valid = denom > 0
    corr = np.zeros(a.shape[1], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr[valid] = numerator[valid] / denom[valid]
    corr[~np.isfinite(corr)] = 0.0
    return corr, valid


def compute_reliability(
    betas: np.ndarray,
    repeat_groups: Mapping[str, Sequence[int]],
) -> np.ndarray:
    n_voxels = betas.shape[1]
    sum_corr = np.zeros(n_voxels, dtype=np.float64)
    contrib = np.zeros(n_voxels, dtype=np.int32)

    max_repeats = max(len(indices) for indices in repeat_groups.values())
    for rep_a in range(max_repeats - 1):
        for rep_b in range(rep_a + 1, max_repeats):
            trial_pairs = [
                (indices[rep_a], indices[rep_b])
                for indices in repeat_groups.values()
                if len(indices) > rep_b
            ]
            if not trial_pairs:
                continue
            trials_a = np.vstack([betas[i, :] for i, _ in trial_pairs])
            trials_b = np.vstack([betas[j, :] for _, j in trial_pairs])
            corr, valid = voxelwise_corr(trials_a, trials_b)
            sum_corr += corr
            contrib += valid.astype(np.int32)

    if not np.any(contrib):
        raise ValueError("Unable to compute reliability; all voxels lacked variance across repeats.")

    reliability = np.divide(
        sum_corr,
        contrib,
        out=np.zeros_like(sum_corr),
        where=contrib > 0,
    )
    return reliability


def threshold_reliability(
    reliability: np.ndarray,
    threshold: float | None,
    quantile: float | None,
) -> Tuple[np.ndarray, float | None]:
    if threshold is not None:
        mask = reliability >= threshold
        return mask, threshold
    if quantile is not None:
        derived = float(np.quantile(reliability, quantile))
        mask = reliability >= derived
        return mask, derived
    return (reliability > 0), None



def format_session_suffix(sessions: Sequence[str]) -> str:
    processed: List[str] = []
    all_prefixed = True
    for ses in sessions:
        if ses.startswith('ses-'):
            processed.append(ses.split('ses-', 1)[1] or ses)
        else:
            processed.append(ses)
            all_prefixed = False
    if not processed:
        return 'sessions'
    if all_prefixed:
        return 'ses-' + '-'.join(processed)
    return '-'.join(processed)


def save_numpy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_union_nifti(
    union_mask: np.ndarray,
    mask_img_path: Path,
    output_path: Path,
) -> None:
    if nib is None:
        raise ModuleNotFoundError("nibabel is required to write NIfTI masks. Install nibabel and retry.")
    mask_img = nib.load(str(mask_img_path))
    mask_data = mask_img.get_fdata().astype(bool)
    flat_true_indices = np.flatnonzero(mask_data.ravel())
    if union_mask.size != flat_true_indices.size:
        raise ValueError(
            "Union mask length does not match number of true voxels in base mask: "
            f"{union_mask.size} vs {flat_true_indices.size}."
        )
    flat_mask = np.zeros(mask_data.size, dtype=np.uint8)
    flat_mask[flat_true_indices[union_mask.astype(bool)]] = 1
    reshaped = flat_mask.reshape(mask_data.shape)
    out_img = nib.Nifti1Image(reshaped.astype(np.uint8), affine=mask_img.affine, header=mask_img.header)
    nib.save(out_img, str(output_path))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    output_dir = resolve_output_dir(args, config)
    derivatives_path = Path(config["derivatives_path"])
    design_task = args.design_task or args.task
    design_dir = Path(config["data_path"]) / "events"

    session_results: List[SessionReliability] = []
    for session in args.sessions:
        try:
            image_names = load_image_names(args.subject, session, design_task, design_dir)
        except Exception as exc:
            raise RuntimeError(f"Failed to load design metadata for {session}: {exc}") from exc

        repeat_groups = build_repeat_groups(image_names, args.test_pattern)
        kept_images = len(repeat_groups)
        kept_trials = sum(len(v) for v in repeat_groups.values())
        skipped_trials = len(image_names) - kept_trials
        print(f"[{session}] usable repeats: {kept_images} images, {kept_trials} trials; skipped {skipped_trials}")

        try:
            beta_file = find_beta_file(derivatives_path, args.subject, session, args.task, args.beta_pattern)
        except FileNotFoundError:
            if args.allow_missing:
                print(f"[warn] Skipping {session}: betas not found")
                continue
            raise
        betas = load_betas(beta_file, expected_trials=len(image_names))
        reliability = compute_reliability(betas, repeat_groups)
        mask, resolved_thresh = threshold_reliability(reliability, args.threshold, args.quantile)

        reliability_path = output_dir / f"{args.subject}_{session}_task-{args.task}_reliability.npy"
        mask_path = output_dir / f"{args.subject}_{session}_task-{args.task}_reliability-mask.npy"
        save_numpy(reliability, reliability_path)
        save_numpy(mask.astype(np.uint8), mask_path)
        session_results.append(SessionReliability(session, reliability, mask, resolved_thresh))
        thresh_msg = f"threshold={resolved_thresh:.3f}" if resolved_thresh is not None else "no threshold"
        print(f"[{session}] reliability saved to {reliability_path.name} ({thresh_msg})")

    if not session_results:
        raise RuntimeError("No sessions were processed successfully; nothing to union.")

    union_mask = np.logical_or.reduce([res.mask.astype(bool) for res in session_results])
    processed_sessions = [res.session for res in session_results]
    session_suffix = format_session_suffix(processed_sessions)
    union_name = f"{args.name_prefix}_{session_suffix}"
    union_path = output_dir / f"{union_name}.npy"
    save_numpy(union_mask.astype(np.uint8), union_path)
    processed_str = ','.join(processed_sessions)
    print(
        f"Union mask saved to {union_path.name} (sessions={processed_str} voxels={int(union_mask.sum())})"
    )

    if args.mask_img:
        mask_img_path = Path(args.mask_img)
        if not mask_img_path.is_file():
            raise FileNotFoundError(f"mask image not found: {mask_img_path}")
        nifti_path = output_dir / f"{union_name}.nii.gz"
        save_union_nifti(union_mask.astype(bool), mask_img_path, nifti_path)
        print(f"Union mask NIfTI saved to {nifti_path.name}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
