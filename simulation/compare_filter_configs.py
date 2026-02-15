"""
Benchmark multiple filter configurations against ground-truth filtered data.

Applies each candidate filter pipeline to the raw signal, computes per-channel
correlation with the ground truth, and selects the best match.  Produces three
plots:
  1. Overlay of raw / ground truth / best filter (5 s window)
  2. Running RMSE divergence (1 s sliding windows)
  3. Point-wise difference (best filter − ground truth)

Usage:
    python -m simulation.compare_filter_configs
    python simulation/compare_filter_configs.py
    python simulation/compare_filter_configs.py --seconds 60
"""

import sys
import os
import argparse

import numpy as np
from scipy.signal import butter, sosfiltfilt

# ── Ensure project root is on the path ────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _project_root)


# -------------------------------------------------------------------
# Filter helpers (all SOS for numerical stability)
# -------------------------------------------------------------------

def bandpass(sig, fs, lo, hi, order=4):
    nyq = fs / 2.0
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, sig)


def highpass(sig, fs, cutoff, order=4):
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="high", output="sos")
    return sosfiltfilt(sos, sig)


def lowpass(sig, fs, cutoff, order=4):
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, sig)


def notch(sig, fs, center, width=2.0, order=4):
    nyq = fs / 2.0
    lo = (center - width / 2.0) / nyq
    hi = (center + width / 2.0) / nyq
    sos = butter(order, [lo, hi], btype="bandstop", output="sos")
    return sosfiltfilt(sos, sig)


# -------------------------------------------------------------------
# Candidate filter pipelines
# -------------------------------------------------------------------

CANDIDATES = {
    "bandpass 0.5-30 Hz":            lambda s, fs: bandpass(s, fs, 0.5, 30),
    "bandpass 1-30 Hz":              lambda s, fs: bandpass(s, fs, 1.0, 30),
    "bandpass 0.1-30 Hz":            lambda s, fs: bandpass(s, fs, 0.1, 30),
    "bandpass 0.5-45 Hz":            lambda s, fs: bandpass(s, fs, 0.5, 45),
    "bandpass 1-45 Hz":              lambda s, fs: bandpass(s, fs, 1.0, 45),
    "HP 0.5 + LP 30":               lambda s, fs: lowpass(highpass(s, fs, 0.5), fs, 30),
    "HP 1.0 + LP 30":               lambda s, fs: lowpass(highpass(s, fs, 1.0), fs, 30),
    "HP 0.5 + LP 30 + notch 60":    lambda s, fs: notch(lowpass(highpass(s, fs, 0.5), fs, 30), fs, 60),
    "HP 0.1 + LP 30":               lambda s, fs: lowpass(highpass(s, fs, 0.1), fs, 30),
    "bandpass 0.5-30 ord2":          lambda s, fs: bandpass(s, fs, 0.5, 30, order=2),
    "bandpass 0.5-30 ord6":          lambda s, fs: bandpass(s, fs, 0.5, 30, order=6),
}


# -------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------

def score_candidate(name, filt_fn, raw, truth, fs, crop_samples=0):
    """Apply filter to all channels, crop transient, return correlation metrics."""
    n_samples, n_ch = raw.shape
    out = np.empty_like(raw)
    for ch in range(n_ch):
        out[:, ch] = filt_fn(raw[:, ch], fs)

    out_c = out[crop_samples:]
    truth_c = truth[crop_samples:]

    corrs = []
    for ch in range(n_ch):
        c = np.corrcoef(out_c[:, ch], truth_c[:, ch])[0, 1]
        corrs.append(c)
    return np.mean(corrs), corrs, out


def running_divergence(ours, truth, fs, window_sec=1.0):
    """Compute RMSE in sliding 1-s windows → (times, per_ch_rmse)."""
    w = int(window_sec * fs)
    n = ours.shape[0]
    n_ch = ours.shape[1]
    steps = n // w
    times = np.arange(steps) * window_sec + window_sec / 2
    rmses = np.zeros((steps, n_ch))
    for i in range(steps):
        s, e = i * w, (i + 1) * w
        for ch in range(n_ch):
            diff = ours[s:e, ch] - truth[s:e, ch]
            rmses[i, ch] = np.sqrt(np.mean(diff ** 2))
    return times, rmses


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main(raw_path, filtered_path, fs, seconds, crop_sec):
    print("Loading CSVs …")
    raw_full = np.loadtxt(raw_path, delimiter=",")
    truth_full = np.loadtxt(filtered_path, delimiter=",")
    n_ch = raw_full.shape[1]
    print(f"  Raw: {raw_full.shape}   Truth: {truth_full.shape}   fs={fs} Hz")

    n_use = min(raw_full.shape[0], int(seconds * fs))
    raw = raw_full[:n_use]
    truth = truth_full[:n_use]

    crop_samples = int(crop_sec * fs)
    print(f"  Using first {n_use / fs:.1f}s ({n_use:,} samples)")
    print(f"  Cropping first {crop_sec:.1f}s ({crop_samples} samples) of filter transient")
    print(f"  Effective comparison window: {crop_sec:.1f}s – {n_use / fs:.1f}s\n")

    # ── Score every candidate ─────────────────────────────────────────
    print("=" * 78)
    print(f"  {'Filter':<32} {'MeanCorr':>9}   Ch0     Ch1     Ch2     Ch3")
    print("-" * 78)

    results = {}
    for name, fn in CANDIDATES.items():
        try:
            mc, corrs, fdata = score_candidate(name, fn, raw, truth, fs,
                                               crop_samples=crop_samples)
            results[name] = (mc, corrs, fdata)
            ch_str = "  ".join(f"{c:+.4f}" for c in corrs)
            print(f"  {name:<32} {mc:>+9.4f}   {ch_str}")
        except Exception as e:
            print(f"  {name:<32}  ERROR: {e}")

    best_name = max(results, key=lambda k: results[k][0])
    best_corr, best_ch_corrs, best_data = results[best_name]
    print("-" * 78)
    print(f"  BEST: {best_name}   (mean corr = {best_corr:+.4f})")
    print("=" * 78)

    # ── Detailed metrics ──────────────────────────────────────────────
    bd = best_data[crop_samples:]
    tr = truth[crop_samples:]

    print(f"\n  Detailed metrics for best filter (after {crop_sec}s crop):")
    print(f"  {'Ch':<5} {'MAE':>10} {'RMSE':>10} {'MaxErr':>10} {'Corr':>10}")
    print("  " + "-" * 50)
    for ch in range(n_ch):
        diff = bd[:, ch] - tr[:, ch]
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        mx = np.max(np.abs(diff))
        print(f"  {ch:<5} {mae:>10.6f} {rmse:>10.6f} {mx:>10.6f} {best_ch_corrs[ch]:>+10.6f}")

    # ── Plots ─────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    raw_c = raw[crop_samples:]
    best_c = best_data[crop_samples:]
    truth_c = truth[crop_samples:]
    n_cropped = raw_c.shape[0]
    plot_samp = min(n_cropped, int(5 * fs))
    tp = crop_sec + np.arange(plot_samp) / fs

    # Fig 1: overlay
    fig1, axes1 = plt.subplots(n_ch, 1, figsize=(16, 2.8 * n_ch), sharex=True)
    if n_ch == 1:
        axes1 = [axes1]
    fig1.suptitle(f"Best: {best_name}  (5 s after {crop_sec}s crop)", fontsize=13)
    for ch in range(n_ch):
        ax = axes1[ch]
        ax.plot(tp, raw_c[:plot_samp, ch],   lw=0.4, alpha=0.4, color="gray",  label="Raw")
        ax.plot(tp, truth_c[:plot_samp, ch],  lw=0.9, color="green",            label="Ground Truth")
        ax.plot(tp, best_c[:plot_samp, ch],   lw=0.9, color="red", ls="--",     label="Ours")
        ax.set_ylabel(f"Ch {ch} (uV)", fontsize=9)
        ax.grid(True, alpha=0.2)
        if ch == 0:
            ax.legend(fontsize=8, loc="upper right")
    axes1[-1].set_xlabel("Time (s)")
    fig1.tight_layout()

    # Fig 2: running RMSE
    div_t, div_rmse = running_divergence(best_c, truth_c, fs, window_sec=1.0)
    div_t_raw, div_rmse_raw = running_divergence(raw_c, truth_c, fs, window_sec=1.0)
    div_t += crop_sec
    div_t_raw += crop_sec

    fig2, axes2 = plt.subplots(n_ch, 1, figsize=(16, 2.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes2 = [axes2]
    fig2.suptitle(f"Running RMSE vs Ground Truth (1 s windows, after {crop_sec}s crop)", fontsize=13)
    for ch in range(n_ch):
        ax = axes2[ch]
        ax.plot(div_t_raw, div_rmse_raw[:, ch], lw=1, color="gray", alpha=0.6, label="Raw vs Truth")
        ax.plot(div_t, div_rmse[:, ch],          lw=1.2, color="red",            label="Ours vs Truth")
        ax.set_ylabel(f"Ch {ch}\nRMSE", fontsize=9)
        ax.grid(True, alpha=0.2)
        if ch == 0:
            ax.legend(fontsize=8, loc="upper right")
    axes2[-1].set_xlabel("Time (s)")
    fig2.tight_layout()

    # Fig 3: point-wise difference
    fig3, axes3 = plt.subplots(n_ch, 1, figsize=(16, 2.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes3 = [axes3]
    fig3.suptitle(f"Difference: Ours − Ground Truth (after {crop_sec}s crop)", fontsize=13)
    t_full = crop_sec + np.arange(n_cropped) / fs
    for ch in range(n_ch):
        ax = axes3[ch]
        diff = best_c[:, ch] - truth_c[:, ch]
        ax.plot(t_full, diff, lw=0.3, color="purple")
        ax.axhline(0, color="black", lw=0.3)
        ax.set_ylabel(f"Ch {ch}\ndiff", fontsize=9)
        ax.grid(True, alpha=0.2)
    axes3[-1].set_xlabel("Time (s)")
    fig3.tight_layout()

    print("\n  Close the plot windows to exit.")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark filter configurations against ground-truth filtered data",
    )
    parser.add_argument("--raw", default="../10_raw.csv")
    parser.add_argument("--filtered", default="../10_filtered.csv")
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="Seconds to compare (default: 30, -1 = all)")
    parser.add_argument("--crop", type=float, default=3.0,
                        help="Seconds to crop from start to skip filter transient (default: 3)")
    args = parser.parse_args()
    secs = args.seconds if args.seconds > 0 else float("inf")
    main(args.raw, args.filtered, args.fs, secs, args.crop)
