"""
Simulate a live EEG data stream by replaying a raw CSV file in packets.

Loads a CSV with an arbitrary number of channels (auto-detected from
the file) and feeds it to a callback one packet at a time, paced at
the real sampling rate (adjustable with a speed multiplier).
Channels are labelled 0, 1, 2, 3, …

Usage:
    # Standalone test — just prints packet info
    python simulate_stream.py

    # Or import and wire up to your pipeline:
    #   from simulate_stream import StreamSimulator
    #   sim = StreamSimulator("../10_raw.csv", fs=200, packet_size=10)
    #   sim.run(callback=my_pipeline.run, speed=5.0)
"""

import time
import argparse
import numpy as np


class StreamSimulator:
    """Replay a CSV file as a simulated real-time data stream."""

    def __init__(
        self,
        csv_path: str,
        fs: float = 200.0,
        packet_size: int = 1,
    ):
        """
        Args:
            csv_path:    Path to a headerless CSV with N columns (auto-detected).
                         Channels are labelled 0, 1, 2, …, N-1.
            fs:          Sampling rate of the data in Hz (default 200).
            packet_size: Number of samples per packet (default 1 = sample-by-sample).
                         Increase to simulate chunked delivery (e.g. 10 or 20).
        """
        self.csv_path = csv_path
        self.fs = fs
        self.packet_size = packet_size
        self.data = np.loadtxt(csv_path, delimiter=",")

        # Handle single-column CSV edge case
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)

        self.n_samples, self.n_channels = self.data.shape
        duration_s = self.n_samples / fs
        ch_labels = ", ".join(str(i) for i in range(self.n_channels))
        print(f"Loaded {csv_path}")
        print(f"  {self.n_samples:,} samples x {self.n_channels} channels [{ch_labels}]")
        print(f"  Duration: {duration_s:.1f}s ({duration_s / 60:.1f} min) at {fs} Hz")
        print(f"  Packet size: {packet_size} sample(s)")
        print()

    def run(
        self,
        callback=None,
        speed: float = 1.0,
        max_seconds: float = None,
        verbose: bool = True,
    ):
        """Stream the data, calling *callback* for every packet.

        Args:
            callback:    Function called with each packet: callback(packet)
                         where packet is an ndarray of shape (packet_size, n_channels).
                         If None, packets are just printed / counted.
            speed:       Playback speed multiplier (default 1.0 = real-time).
                         Use e.g. 10.0 to replay 10x faster, or 0 for no delay
                         (as fast as possible).
            max_seconds: Optional cap on how many seconds of data to stream
                         (in *data* time, not wall-clock time).  None = all.
            verbose:     Print progress every second of data time.
        """
        n_total = self.data.shape[0]
        if max_seconds is not None:
            n_total = min(n_total, int(max_seconds * self.fs))

        delay_per_packet = (self.packet_size / self.fs) / speed if speed > 0 else 0
        n_packets = n_total // self.packet_size
        report_interval = max(1, int(self.fs / self.packet_size))  # ~every 1s of data

        print(f"Streaming {n_packets:,} packets  |  speed: {speed}x  |  "
              f"delay/packet: {delay_per_packet * 1000:.2f} ms")
        print("-" * 60)

        t_start = time.perf_counter()
        samples_sent = 0

        try:
            for i in range(n_packets):
                start_idx = i * self.packet_size
                end_idx = start_idx + self.packet_size
                packet = self.data[start_idx:end_idx]  # shape (packet_size, n_channels)

                if callback is not None:
                    callback(packet)

                samples_sent += self.packet_size

                if verbose and (i + 1) % report_interval == 0:
                    data_time = samples_sent / self.fs
                    wall_time = time.perf_counter() - t_start
                    print(
                        f"  [{data_time:7.1f}s data | {wall_time:7.1f}s wall]  "
                        f"packet {i + 1:>8,} / {n_packets:,}  "
                        f"samples sent: {samples_sent:>10,}"
                    )

                if delay_per_packet > 0:
                    time.sleep(delay_per_packet)

        except KeyboardInterrupt:
            print("\n--- Interrupted by user ---")

        elapsed = time.perf_counter() - t_start
        print("-" * 60)
        print(f"Done. Sent {samples_sent:,} samples in {elapsed:.2f}s wall-clock time.")
        print(f"  Data time covered: {samples_sent / self.fs:.1f}s")
        print(f"  Effective speed:   {(samples_sent / self.fs) / elapsed:.1f}x real-time")


# ── Default callback for standalone testing ──────────────────────────
def _print_callback(packet):
    """Simple callback that prints the first sample of each packet."""
    vals = ", ".join(f"{i}:{v:+.4f}" for i, v in enumerate(packet[0]))
    print(f"  >> [{vals}]")


# ── CLI entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate live EEG data stream")
    parser.add_argument(
        "--csv", type=str, default="../10_raw.csv",
        help="Path to the raw CSV file (default: ../10_raw.csv)",
    )
    parser.add_argument(
        "--fs", type=float, default=200.0,
        help="Sampling rate in Hz (default: 200)",
    )
    parser.add_argument(
        "--packet-size", type=int, default=10,
        help="Samples per packet (default: 10)",
    )
    parser.add_argument(
        "--speed", type=float, default=10.0,
        help="Playback speed multiplier (default: 10x, use 0 for max speed)",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=5.0,
        help="Max seconds of data to stream (default: 5s, use -1 for all)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-packet printing (still shows progress)",
    )
    args = parser.parse_args()

    max_s = None if args.max_seconds < 0 else args.max_seconds
    cb = None if args.quiet else _print_callback

    sim = StreamSimulator(args.csv, fs=args.fs, packet_size=args.packet_size)
    sim.run(callback=cb, speed=args.speed, max_seconds=max_s)
