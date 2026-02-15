"""
End-to-end test: simulated EEG stream → PreprocessingPipeline.

The StreamSimulator delivers packets of raw voltage (packet_size × n_channels).
This script accumulates those packets into windows shaped for the pipeline:

    (n_samples, 1 + n_channels)
    column 0      = timestamps (seconds)
    columns 1…N   = voltage channels

Each full window is downsampled to the pipeline's target_fs (100 Hz default),
then lowpass + bandstop filters are applied.  Raw vs filtered results are
plotted at the end.

Usage:
    python -m simulation.test_stream_pipeline
    python simulation/test_stream_pipeline.py
    python simulation/test_stream_pipeline.py --csv ../10_raw.csv --fs 200 --window 1.0 --speed 0
"""

import sys
import os
import argparse

import numpy as np

# ── Ensure sibling packages are importable ────────────────────────────
_project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _project_root)

from simulation.simulate_eeg_stream import StreamSimulator
from analysis.PreprocessingPipeline import PreprocessingPipeline


class StreamToPipeline:
    """Accumulate streamed packets into windows and run preprocessing."""

    def __init__(self, fs: float, window_seconds: float, target_fs: float = 100.0):
        """
        Args:
            fs:             Sampling rate of the incoming data (Hz).
            window_seconds: How many seconds of data per processing window.
            target_fs:      Target sampling rate after downsampling (Hz).
        """
        self.fs = fs
        self.window_size = int(window_seconds * fs)
        self.pipeline = PreprocessingPipeline(target_fs=target_fs)

        self.buffer = None
        self.sample_index = 0
        self.window_count = 0

        self.raw_windows = []
        self.filtered_windows = []

    def on_packet(self, packet: np.ndarray):
        """Callback for StreamSimulator — receives (packet_size, n_channels)."""
        if self.buffer is None:
            n_channels = packet.shape[1]
            self.buffer = np.empty((self.window_size, n_channels), dtype=np.float64)
            self.buf_pos = 0
            print(f"\n  Pipeline initialized: {n_channels} channels, "
                  f"window={self.window_size} samples ({self.window_size / self.fs:.2f}s), "
                  f"target_fs={self.pipeline.target_fs} Hz\n")

        n_new = packet.shape[0]
        space = self.window_size - self.buf_pos

        if n_new <= space:
            self.buffer[self.buf_pos:self.buf_pos + n_new] = packet
            self.buf_pos += n_new
            self.sample_index += n_new
        else:
            self.buffer[self.buf_pos:self.buf_pos + space] = packet[:space]
            self.buf_pos = self.window_size
            self.sample_index += space
            self._process_window()

            leftover = packet[space:]
            self.buf_pos = leftover.shape[0]
            self.buffer[:self.buf_pos] = leftover
            self.sample_index += leftover.shape[0]

        if self.buf_pos == self.window_size:
            self._process_window()

    def _process_window(self):
        """Run the full preprocessing pipeline on the current window."""
        self.window_count += 1
        n_channels = self.buffer.shape[1]

        t_start = (self.sample_index - self.window_size) / self.fs
        times = t_start + np.arange(self.window_size) / self.fs
        window_data = np.column_stack([times, self.buffer[:self.window_size]])

        # Stage 1: Downsample
        downsampled = self.pipeline.downsample_window(window_data)

        # Stage 2: Filters (per-channel on the downsampled data)
        ds_fs = self.pipeline.target_fs
        n_ds_samples = downsampled.shape[0]
        filtered = downsampled.copy()

        for ch in range(1, n_channels + 1):
            sig = filtered[:, ch]
            if n_ds_samples >= 13:  # minimum for order-4 filtfilt
                sig = self.pipeline.lowpass_blink_filter(sig, ds_fs)
                sig = self.pipeline.bandstop_sweat_filter(sig, ds_fs)
            filtered[:, ch] = sig

        self.raw_windows.append(window_data.copy())
        self.filtered_windows.append(filtered.copy())

        print(f"  Window {self.window_count:>4}  |  "
              f"t=[{window_data[0, 0]:7.2f}s - {window_data[-1, 0]:7.2f}s]  |  "
              f"raw {window_data.shape} -> ds {downsampled.shape} -> filtered {filtered.shape}")

        if self.window_count <= 3:
            print(f"    First row (time + channels): {filtered[0]}")
            print(f"    Last  row (time + channels): {filtered[-1]}")
            print()

        self.buf_pos = 0

    def flush(self):
        """Process any remaining samples in the buffer (partial window)."""
        if self.buffer is not None and self.buf_pos >= 2:
            old_ws = self.window_size
            self.window_size = self.buf_pos
            self._process_window()
            self.window_size = old_ws
            print("  (flushed partial window)")


def plot_raw_vs_filtered(raw_windows, filtered_windows):
    """Plot raw vs filtered time-series for every channel, side by side."""
    import matplotlib.pyplot as plt

    raw = np.vstack(raw_windows)
    filt = np.vstack(filtered_windows)

    n_channels = raw.shape[1] - 1
    raw_t = raw[:, 0]
    filt_t = filt[:, 0]

    fig, axes = plt.subplots(n_channels, 2,
                             figsize=(16, 3 * n_channels),
                             sharex="col")
    if n_channels == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle("Before (Raw) vs After (Filtered + Downsampled)", fontsize=14, y=1.0)

    for ch in range(n_channels):
        ax_raw = axes[ch][0]
        ax_raw.plot(raw_t, raw[:, ch + 1], linewidth=0.4, color="steelblue")
        ax_raw.set_ylabel(f"Ch {ch} (uV)", fontsize=9)
        ax_raw.grid(True, alpha=0.3)
        if ch == 0:
            ax_raw.set_title("Raw (before)", fontsize=11)

        ax_filt = axes[ch][1]
        ax_filt.plot(filt_t, filt[:, ch + 1], linewidth=0.4, color="tomato")
        ax_filt.grid(True, alpha=0.3)
        if ch == 0:
            ax_filt.set_title("Filtered + Downsampled (after)", fontsize=11)

    axes[-1][0].set_xlabel("Time (s)", fontsize=10)
    axes[-1][1].set_xlabel("Time (s)", fontsize=10)

    fig.tight_layout()
    print("  Close the plot window to exit.")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test: simulated EEG stream -> PreprocessingPipeline",
    )
    parser.add_argument("--csv", type=str, default="../10_raw.csv",
                        help="Path to raw CSV (default: ../10_raw.csv)")
    parser.add_argument("--fs", type=float, default=200.0,
                        help="Sampling rate in Hz (default: 200)")
    parser.add_argument("--window", type=float, default=1.0,
                        help="Window size in seconds (default: 1.0)")
    parser.add_argument("--speed", type=float, default=0,
                        help="Playback speed (default: 0 = max speed)")
    parser.add_argument("--max-seconds", type=float, default=10.0,
                        help="Max seconds of data (default: 10, -1 = all)")
    args = parser.parse_args()

    max_s = None if args.max_seconds < 0 else args.max_seconds

    handler = StreamToPipeline(fs=args.fs, window_seconds=args.window)
    sim = StreamSimulator(args.csv, fs=args.fs, packet_size=10)
    sim.run(callback=handler.on_packet, speed=args.speed, max_seconds=max_s)
    handler.flush()

    print("\n" + "=" * 70)
    print(f"  Done.  Processed {handler.window_count} windows.")
    print("=" * 70)

    if handler.raw_windows and handler.filtered_windows:
        plot_raw_vs_filtered(handler.raw_windows, handler.filtered_windows)
