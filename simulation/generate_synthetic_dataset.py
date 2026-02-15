"""
Generate a synthetic EEG/EMG dataset for motor-imagery classification.

Creates realistic multi-channel signals for five movement classes:
  - rest              : relaxed baseline with prominent alpha rhythm
  - motor_imagery     : imagined movement (mu/alpha suppression, beta modulation)
  - wrist_flex_ext    : alternating wrist flexion/extension EMG bursts
  - grip_release      : sustained grip followed by release decay
  - cocontraction     : simultaneous antagonist muscle activation

Output structure (organized by class):
    simulation/dataset/
    ├── manifest.csv            Trial index with paths and metadata
    ├── class_map.json          Numeric ID → class name mapping
    ├── rest/                   NPZ data + TSV events for rest trials
    ├── motor_imagery/          NPZ data + TSV events for imagery trials
    ├── wrist_flex_ext/         NPZ data + TSV events for flex/ext trials
    ├── grip_release/           NPZ data + TSV events for grip trials
    └── cocontraction/          NPZ data + TSV events for cocontraction trials

Usage:
    python -m simulation.generate_synthetic_dataset
    python simulation/generate_synthetic_dataset.py
    python simulation/generate_synthetic_dataset.py --subjects 5 --trials 20
"""

import os
import json
import csv
import argparse

import numpy as np

# -------------------------------------------------------------------
# Global signal parameters
# -------------------------------------------------------------------
SAMPLING_RATE_HZ = 200
TRIAL_DURATION_S = 10.0
N_SAMPLES_PER_TRIAL = int(SAMPLING_RATE_HZ * TRIAL_DURATION_S)

CHANNEL_NAMES = ["ch0_eeg", "ch1_eeg", "ch2_eeg", "ch3_emg"]
CHANNEL_UNITS = ["uV", "uV", "uV", "mV"]

CLASS_NAMES = ("rest", "motor_imagery", "wrist_flex_ext", "grip_release", "cocontraction")

DEFAULT_SEED = 12345
RNG = np.random.default_rng(DEFAULT_SEED)


# -------------------------------------------------------------------
# Signal-building helpers
# -------------------------------------------------------------------

def band_limited_noise(n: int, fs: float, f_lo: float, f_hi: float, rng) -> np.ndarray:
    """Create real-valued band-limited noise using FFT masking."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    X *= mask.astype(float)
    y = np.fft.irfft(X, n=n)
    y /= (np.std(y) + 1e-12)
    return y


def pinkish_noise(n: int, rng) -> np.ndarray:
    """Approximate 1/f noise by filtering white noise in the frequency domain."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    scale = 1.0 / np.maximum(freqs, 1.0)  # avoid div-by-zero at DC
    X *= scale
    y = np.fft.irfft(X, n=n)
    y /= (np.std(y) + 1e-12)
    return y


def smooth_envelope(n: int, fs: float, segments: list) -> np.ndarray:
    """Build a smooth amplitude envelope from (start_s, end_s, level) segments.

    Uses cosine-ramped Hann smoothing (~50 ms window) to avoid sharp edges.
    """
    t = np.arange(n) / fs
    env = np.zeros(n, dtype=float)
    for (a, b, level) in segments:
        m = (t >= a) & (t <= b)
        env[m] = np.maximum(env[m], level)

    win_len = max(5, int(0.05 * fs))
    w = np.hanning(win_len)
    w /= w.sum()
    return np.convolve(env, w, mode="same")


def add_line_noise(x: np.ndarray, fs: float, freq: float = 60.0, amp: float = 0.02) -> np.ndarray:
    """Add simulated power-line interference."""
    t = np.arange(len(x)) / fs
    return x + amp * np.sin(2 * np.pi * freq * t)


# -------------------------------------------------------------------
# Per-class trial generators
# -------------------------------------------------------------------

def generate_trial(class_name: str, fs: float, duration_s: float, rng) -> tuple:
    """Synthesize one trial of (4-channel data, event list) for the given class.

    Channel layout:
        ch0_eeg — frontal / motor-cortex EEG (µV)
        ch1_eeg — central / sensorimotor EEG (µV)
        ch2_eeg — parietal / posterior EEG (µV)
        ch3_emg — forearm muscle EMG (mV)

    Returns:
        data:   np.ndarray of shape (4, n_samples) — [eeg0, eeg1, eeg2, emg]
        events: list of (onset_s, duration_s, label) tuples
    """
    n = int(fs * duration_s)
    t = np.arange(n) / fs

    # ── Base signals ──────────────────────────────────────────────
    # Three EEG channels: independent pinkish (1/f) backgrounds
    eeg0 = 0.6 * pinkish_noise(n, rng)   # frontal
    eeg1 = 0.6 * pinkish_noise(n, rng)   # central
    eeg2 = 0.6 * pinkish_noise(n, rng)   # parietal

    # One EMG channel: low-level broadband baseline tone
    emg = 0.15 * band_limited_noise(n, fs, 20, 120, rng)

    events = []

    # ── Class-specific shaping ────────────────────────────────────
    if class_name == "rest":
        # Strong alpha (8-12 Hz) across all EEG, strongest in parietal;
        # EMG stays at baseline.
        alpha_f = rng.uniform(8.5, 11.5)
        eeg0 += 0.7 * np.sin(2 * np.pi * alpha_f * t)
        eeg1 += 0.9 * np.sin(2 * np.pi * (alpha_f + rng.uniform(-0.3, 0.3)) * t)
        eeg2 += 1.2 * np.sin(2 * np.pi * (alpha_f + rng.uniform(-0.2, 0.2)) * t)
        events.append((0.0, duration_s, "rest"))

    elif class_name == "motor_imagery":
        # Mu/alpha suppression mid-trial in central + frontal;
        # beta rebound in parietal; EMG stays quiet.
        alpha_f = rng.uniform(9.0, 12.0)
        beta_f = rng.uniform(18.0, 26.0)
        alpha = np.sin(2 * np.pi * alpha_f * t)
        beta = np.sin(2 * np.pi * beta_f * t)

        # Envelope: strong alpha early/late, suppressed in the middle
        env = smooth_envelope(n, fs, [(0.0, 3.0, 1.0), (3.0, 7.0, 0.35), (7.0, 10.0, 1.0)])
        eeg0 += 0.8 * env * alpha + 0.20 * beta
        eeg1 += 1.0 * env * np.sin(2 * np.pi * (alpha_f + 0.4) * t) + 0.30 * beta
        eeg2 += 0.6 * env * np.sin(2 * np.pi * (alpha_f - 0.3) * t) + 0.35 * np.sin(2 * np.pi * (beta_f - 1.2) * t)

        emg *= 0.5  # EMG stays quiet during imagery
        events.append((0.0, 3.0, "rest"))
        events.append((3.0, 4.0, "imagery"))
        events.append((7.0, 3.0, "rest"))

    elif class_name == "wrist_flex_ext":
        # Alternating EMG bursts (flex/extend on single muscle channel);
        # EEG shows beta desync during movement windows.
        cycles = [
            (1.0, 2.0, "flex"),    (2.5, 3.5, "extend"),
            (4.0, 5.0, "flex"),    (5.5, 6.5, "extend"),
            (7.0, 8.0, "flex"),    (8.5, 9.5, "extend"),
        ]
        flex_env = smooth_envelope(n, fs, [(a, b, 1.0) for a, b, lab in cycles if lab == "flex"])
        ext_env  = smooth_envelope(n, fs, [(a, b, 1.0) for a, b, lab in cycles if lab == "extend"])
        combined_env = np.clip(flex_env + ext_env, 0, 1)

        # EMG: alternating bursts with slightly different amplitudes
        emg += 1.2 * flex_env * band_limited_noise(n, fs, 25, 150, rng)
        emg += 0.9 * ext_env  * band_limited_noise(n, fs, 25, 150, rng)

        # EEG: beta desynchronization during movement
        beta_f = rng.uniform(18, 24)
        eeg0 += 0.25 * combined_env * np.sin(2 * np.pi * beta_f * t)
        eeg1 += 0.30 * combined_env * np.sin(2 * np.pi * (beta_f + 1.0) * t)
        eeg2 += 0.15 * combined_env * np.sin(2 * np.pi * (beta_f - 0.5) * t)

        events.append((0.0, 1.0, "rest"))
        for a, b, lab in cycles:
            events.append((a, b - a, lab))
        events.append((9.5, 0.5, "rest"))

    elif class_name == "grip_release":
        # Sustained grip → release on a single EMG channel;
        # EEG shows movement-related beta modulation.
        grip    = smooth_envelope(n, fs, [(2.0, 6.0, 1.0)])
        release = smooth_envelope(n, fs, [(6.0, 8.5, 0.6)])
        emg_env = np.clip(grip + release, 0, 1.0)

        emg += 1.5 * emg_env * band_limited_noise(n, fs, 30, 180, rng)

        beta_f = rng.uniform(16, 24)
        eeg0 += 0.20 * emg_env * np.sin(2 * np.pi * beta_f * t)
        eeg1 += 0.25 * emg_env * np.sin(2 * np.pi * (beta_f + 0.8) * t)
        eeg2 += 0.10 * emg_env * np.sin(2 * np.pi * (beta_f - 0.6) * t)

        events.append((0.0, 2.0, "rest"))
        events.append((2.0, 4.0, "grip"))
        events.append((6.0, 2.5, "release"))
        events.append((8.5, 1.5, "rest"))

    elif class_name == "cocontraction":
        # Sustained co-activation: high EMG baseline + burst;
        # EEG picks up low-freq tension drift.
        base_tone = 0.25 * band_limited_noise(n, fs, 15, 80, rng)
        emg += base_tone

        co = smooth_envelope(n, fs, [(1.5, 8.5, 1.0)])
        emg += 1.1 * co * band_limited_noise(n, fs, 25, 150, rng)

        drift = 0.15 * np.cumsum(rng.standard_normal(n)) / n
        eeg0 += drift
        eeg1 += 0.8 * drift
        eeg2 += 0.5 * drift

        events.append((0.0, 1.5, "rest"))
        events.append((1.5, 7.0, "cocontraction"))
        events.append((8.5, 1.5, "rest"))

    else:
        raise ValueError(f"Unknown class: {class_name}")

    # ── Add mild 60 Hz line noise to all channels ─────────────────
    eeg0 = add_line_noise(eeg0, fs, 60.0, amp=0.03)
    eeg1 = add_line_noise(eeg1, fs, 60.0, amp=0.03)
    eeg2 = add_line_noise(eeg2, fs, 60.0, amp=0.03)
    emg  = add_line_noise(emg,  fs, 60.0, amp=0.01)

    # ── Scale to realistic units (EEG ~µV, EMG ~mV) ──────────────
    eeg0_uV = 20.0 * eeg0
    eeg1_uV = 20.0 * eeg1
    eeg2_uV = 20.0 * eeg2
    emg_mV  = 1.0  * emg

    data = np.vstack([eeg0_uV, eeg1_uV, eeg2_uV, emg_mV]).astype(np.float32)
    return data, events


# -------------------------------------------------------------------
# File I/O
# -------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_events_tsv(path: str, events: list):
    """Write event annotations as a tab-separated file."""
    with open(path, "w", newline="") as f:
        f.write("onset_s\tduration_s\tlabel\n")
        for onset, dur, lab in events:
            f.write(f"{onset:.3f}\t{dur:.3f}\t{lab}\n")


# -------------------------------------------------------------------
# Main dataset generation
# -------------------------------------------------------------------

def generate_dataset(
    out_root: str = os.path.join(os.path.dirname(__file__), "dataset"),
    subjects: int = 3,
    sessions: int = 1,
    trials_per_class: int = 10,
    classes: tuple = CLASS_NAMES,
    seed: int = DEFAULT_SEED,
):
    """Generate the full synthetic dataset on disk.

    Args:
        out_root:         Root directory for the dataset.
        subjects:         Number of synthetic subjects.
        sessions:         Sessions per subject.
        trials_per_class: Trials per class per session.
        classes:          Tuple of class names to generate.
        seed:             Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    _ensure_dir(out_root)

    # Write class map
    class_map = {str(i): c for i, c in enumerate(classes)}
    with open(os.path.join(out_root, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)

    # Create one directory per class
    for c in classes:
        _ensure_dir(os.path.join(out_root, c))

    # Write manifest + data files (organized by class)
    manifest_path = os.path.join(out_root, "manifest.csv")
    with open(manifest_path, "w", newline="") as fman:
        writer = csv.DictWriter(fman, fieldnames=[
            "subject", "session", "trial",
            "class", "class_id",
            "fs", "duration_s",
            "npz_path", "events_path",
            "ch_names",
        ])
        writer.writeheader()

        for s in range(1, subjects + 1):
            sub = f"sub-{s:03d}"
            for ses_i in range(1, sessions + 1):
                ses = f"ses-{ses_i:03d}"

                trial_counter = 0
                for class_id, class_name in enumerate(classes):
                    class_dir = os.path.join(out_root, class_name)

                    for k in range(1, trials_per_class + 1):
                        trial_counter += 1
                        trial = f"{trial_counter:03d}"

                        data, events = generate_trial(class_name, SAMPLING_RATE_HZ, TRIAL_DURATION_S, rng)

                        # Data file → <class>/<sub>_<ses>_trial-NNN_eegemg.npz
                        npz_name = f"{sub}_{ses}_trial-{trial}_eegemg.npz"
                        npz_path = os.path.join(class_dir, npz_name)

                        np.savez_compressed(
                            npz_path,
                            data=data,
                            fs=SAMPLING_RATE_HZ,
                            duration_s=TRIAL_DURATION_S,
                            ch_names=np.array(CHANNEL_NAMES, dtype=object),
                            units=np.array(CHANNEL_UNITS, dtype=object),
                            class_name=class_name,
                            class_id=class_id,
                        )

                        # Events file → <class>/<sub>_<ses>_trial-NNN_events.tsv
                        ev_name = f"{sub}_{ses}_trial-{trial}_events.tsv"
                        ev_path = os.path.join(class_dir, ev_name)
                        _write_events_tsv(ev_path, events)

                        writer.writerow({
                            "subject": sub,
                            "session": ses,
                            "trial": trial,
                            "class": class_name,
                            "class_id": class_id,
                            "fs": SAMPLING_RATE_HZ,
                            "duration_s": TRIAL_DURATION_S,
                            "npz_path": os.path.relpath(npz_path, out_root),
                            "events_path": os.path.relpath(ev_path, out_root),
                            "ch_names": "|".join(CHANNEL_NAMES),
                        })

    print(f"Done — wrote dataset to: {os.path.abspath(out_root)}")
    print(f"  manifest.csv  : {manifest_path}")
    print(f"  class_map.json: {os.path.join(out_root, 'class_map.json')}")
    print(f"  subjects={subjects}, sessions={sessions}, "
          f"trials/class={trials_per_class}, classes={len(classes)}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic EEG/EMG dataset for motor-imagery classification.",
    )
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__), "dataset"),
                        help="Output root directory (default: simulation/dataset)")
    parser.add_argument("--subjects", type=int, default=3,
                        help="Number of subjects (default: 3)")
    parser.add_argument("--sessions", type=int, default=1,
                        help="Sessions per subject (default: 1)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per class per session (default: 10)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    generate_dataset(
        out_root=args.out,
        subjects=args.subjects,
        sessions=args.sessions,
        trials_per_class=args.trials,
        seed=args.seed,
    )
