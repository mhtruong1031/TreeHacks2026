"""
Receive live EEG data from the OpenBCI GUI via UDP.

Collects all incoming samples and, when you press Ctrl+C, displays
matplotlib plots of the full recording:
  - Time-series traces for every channel
  - FFT (frequency vs amplitude) for every channel

Usage:
    python udp_receiver.py                                    # console only
    python udp_receiver.py --plot --fs 250                    # + plots after stop
    python udp_receiver.py --ip 0.0.0.0 --port 12345 --plot --fs 250

Set --ip to 0.0.0.0 to listen on all network interfaces (recommended when
receiving from another machine on the same network).
"""

import socket
import json
import argparse
import time
import sys

import numpy as np


def _flatten_or_wrap(lst):
    """Given a list, return a list of samples (each sample = list of floats).

    If *lst* is a flat list of numbers  → [[n1, n2, ...]]       (1 sample)
    If *lst* is a list of lists         → [[...], [...], ...]   (N samples)
    """
    if not lst:
        return []
    # Nested: list of lists (OpenBCI "timeSeriesRaw" format)
    if isinstance(lst[0], list):
        return [[float(v) for v in row] for row in lst]
    # Flat: single sample
    return [[float(v) for v in lst]]


def parse_packet(raw: bytes):
    """Decode a UDP packet into a list of samples.

    Each sample is a list of per-channel floats.
    Returns (packet_type, samples) where packet_type is a string label
    (e.g. "timeSeriesRaw") or None, and samples is a list of lists:
        [[ch0, ch1, …], [ch0, ch1, …], ...]

    Supports:
      1. OpenBCI JSON  {"type":"timeSeriesRaw", "data":[[...], [...]]}
      2. JSON object with a list-valued key (flat or nested)
      3. Plain JSON array of floats or arrays
      4. Comma-separated text
      5. Space/tab-separated text
    """
    text = raw.decode("utf-8", errors="replace").strip()

    # ── JSON ──────────────────────────────────────────────────────────
    if text.startswith("{") or text.startswith("["):
        try:
            obj = json.loads(text)

            if isinstance(obj, dict):
                pkt_type = obj.get("type", None)
                for key in ("data", "channel_data", "eeg", "values", "samples"):
                    if key in obj and isinstance(obj[key], list):
                        return pkt_type, _flatten_or_wrap(obj[key])
                # Fall back: first list-valued entry
                for v in obj.values():
                    if isinstance(v, list):
                        return pkt_type, _flatten_or_wrap(v)

            if isinstance(obj, list):
                return None, _flatten_or_wrap(obj)

        except (json.JSONDecodeError, ValueError):
            pass  # not valid JSON — try CSV below

    # ── CSV / whitespace-separated ────────────────────────────────────
    for sep in (",", "\t", " "):
        parts = text.split(sep)
        if len(parts) >= 2:
            try:
                row = [float(p) for p in parts if p.strip()]
                return None, [row]
            except ValueError:
                continue

    # Single value
    try:
        return None, [[float(text)]]
    except ValueError:
        return None, None


def pretty_print(sample_num, channels, num_expected=None):
    """Print a single sample's channel values in a readable format."""
    n = len(channels)
    header = f"Sample {sample_num:>8,}  |  {n} ch  |  "
    vals = "  ".join(f"Ch{i}:{v:>+12.4f}" for i, v in enumerate(channels))
    print(header + vals)


def plot_recording(all_samples, fs: float, max_freq: float = 60.0):
    """Show time-series and FFT plots for every channel after recording ends.

    Args:
        all_samples: list of [ch0, ch1, …] lists collected during the run.
        fs:          Sampling rate in Hz.
        max_freq:    Max frequency to display on the FFT x-axis.
    """
    import matplotlib.pyplot as plt

    data = np.array(all_samples)           # shape (n_samples, n_channels)
    n_samples, n_channels = data.shape
    duration = n_samples / fs
    t = np.arange(n_samples) / fs

    print(f"\n  Plotting {n_samples:,} samples x {n_channels} channels "
          f"({duration:.1f}s at {fs} Hz) ...")

    # ── Figure 1: Time-series traces ──────────────────────────────────
    fig1, axes1 = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels),
                               sharex=True)
    if n_channels == 1:
        axes1 = [axes1]
    fig1.suptitle("Time Series — All Channels", fontsize=14, y=1.0)

    for ch in range(n_channels):
        ax = axes1[ch]
        ax.plot(t, data[:, ch], linewidth=0.5)
        ax.set_ylabel(f"Ch {ch}\n(uV)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    axes1[-1].set_xlabel("Time (s)", fontsize=10)
    fig1.tight_layout()

    # ── Figure 2: FFT per channel ─────────────────────────────────────
    cols = min(4, n_channels)
    rows = int(np.ceil(n_channels / cols))
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                               squeeze=False)
    fig2.suptitle("FFT — Frequency vs Amplitude", fontsize=14)

    for ch in range(n_channels):
        r, c = divmod(ch, cols)
        ax = axes2[r][c]

        sig = data[:, ch] - np.mean(data[:, ch])       # remove DC
        window = np.hanning(len(sig))
        sig = sig * window

        fft_vals = np.fft.rfft(sig)
        fft_mag = (2.0 / len(sig)) * np.abs(fft_vals)
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)

        mask = freqs <= max_freq
        ax.plot(freqs[mask], fft_mag[mask], linewidth=0.8)
        ax.set_title(f"Ch {ch}", fontsize=10)
        ax.set_xlabel("Freq (Hz)", fontsize=8)
        ax.set_ylabel("Amplitude (uV)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_channels, rows * cols):
        r, c = divmod(idx, cols)
        axes2[r][c].set_visible(False)

    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    print("  Close the plot windows to exit.")
    plt.show()


def run_receiver(ip: str, port: int, num_channels: int | None, timeout: float,
                 plot: bool = False, fs: float = 250.0, max_freq: float = 60.0):
    """Main loop: bind a UDP socket and print every packet.

    If *plot* is True, store all samples and show plots after Ctrl+C.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))
    sock.settimeout(timeout)

    print("=" * 70)
    print(f"  OpenBCI UDP Receiver")
    print(f"  Listening on {ip}:{port}")
    if num_channels:
        print(f"  Expecting {num_channels} channels")
    else:
        print(f"  Channels: auto-detect from first packet")
    if plot:
        print(f"  Plot after stop: ON  (fs={fs} Hz, max_freq={max_freq} Hz)")
        print(f"  Recording all samples — press Ctrl+C when done to see plots")
    print(f"  Press Ctrl+C to stop")
    print("=" * 70)
    print()

    sample_num = 0
    packet_num = 0
    t_start = None
    detected_channels = num_channels
    detected_type = None
    all_samples = []       # collect every sample when --plot is used

    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                print(f"  (waiting for data on {ip}:{port} ...)")
                continue

            if t_start is None:
                t_start = time.perf_counter()
                print(f"  First packet received from {addr[0]}:{addr[1]}")
                print(f"  Raw bytes ({len(data)}): {data[:200]}{'...' if len(data)>200 else ''}")
                print()

            pkt_type, samples = parse_packet(data)

            if samples is None:
                print(f"  [!] Could not parse packet: {data[:80]}")
                continue

            packet_num += 1

            if detected_channels is None and samples:
                detected_channels = len(samples[0])
                detected_type = pkt_type or "unknown"
                print(f"  Packet type:  {detected_type}")
                print(f"  Channels:     {detected_channels}")
                print(f"  Samples/pkt:  {len(samples)}")
                print("-" * 70)

            for channel_values in samples:
                sample_num += 1

                if plot:
                    all_samples.append(channel_values)

                # Print every sample (or throttle if collecting lots of data)
                if sample_num % 50 == 0 or not plot:
                    pretty_print(sample_num, channel_values, detected_channels)

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - t_start if t_start else 0
        print()
        print("=" * 70)
        print(f"  Stopped.")
        print(f"  Received {packet_num:,} packets / {sample_num:,} samples in {elapsed:.1f}s")
        if elapsed > 0:
            print(f"  Effective rate: {sample_num / elapsed:.1f} samples/s")
        print("=" * 70)
        sock.close()

    # ── Show plots after the run ──────────────────────────────────────
    if plot and len(all_samples) > 0:
        plot_recording(all_samples, fs, max_freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Receive OpenBCI EEG data via UDP and display channel values",
    )
    parser.add_argument(
        "--ip", type=str, default="0.0.0.0",
        help="IP address to bind to (default: 0.0.0.0 = all interfaces)",
    )
    parser.add_argument(
        "--port", type=int, default=12345,
        help="UDP port to listen on (default: 12345)",
    )
    parser.add_argument(
        "--channels", type=int, default=None,
        help="Expected number of channels (default: auto-detect)",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help="Socket timeout in seconds before printing a 'waiting' message (default: 5)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Record all samples and show time-series + FFT plots after Ctrl+C",
    )
    parser.add_argument(
        "--fs", type=float, default=250.0,
        help="Sampling rate in Hz — must match the OpenBCI board rate (default: 250)",
    )
    parser.add_argument(
        "--max-freq", type=float, default=60.0,
        help="Maximum frequency shown on the FFT plot in Hz (default: 60)",
    )
    args = parser.parse_args()

    run_receiver(
        args.ip, args.port, args.channels, args.timeout,
        plot=args.plot, fs=args.fs, max_freq=args.max_freq,
    )
