"""
Simulated data source for CSV file playback.

Loads a CSV file and replays it in real-time (or faster) using a background
thread that pushes packets to a queue. Supports configurable playback speed.
"""

import time
import threading
import queue
import numpy as np
from .base import DataSource


class SimulatedDataSource(DataSource):
    """CSV-based simulated data source with background thread."""

    def __init__(
        self,
        csv_path: str,
        fs: float = 200.0,
        speed: float = 1.0,
        queue_size: int = 1000
    ):
        """
        Args:
            csv_path: Path to headerless CSV file (auto-detects number of channels)
            fs: Sampling rate in Hz (default: 200.0)
            speed: Playback speed multiplier (0 = unlimited, 1 = real-time, 10 = 10x faster)
            queue_size: Max packets in queue (default: 1000)
        """
        self.csv_path = csv_path
        self.fs = fs
        self.speed = speed
        self._running = True
        self._thread_finished = False

        # Load data
        self.data = np.loadtxt(csv_path, delimiter=",")
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)

        self.n_samples, self.n_channels = self.data.shape
        duration_s = self.n_samples / fs

        print(f"Loaded {csv_path}")
        print(f"  {self.n_samples:,} samples x {self.n_channels} channels")
        print(f"  Duration: {duration_s:.1f}s at {fs} Hz")
        print(f"  Playback speed: {speed}x")

        # Queue for producer-consumer pattern
        self.packet_queue = queue.Queue(maxsize=queue_size)

        # Start background thread
        self.producer_thread = threading.Thread(
            target=self._producer,
            daemon=True,
            name="SimulatedDataProducer"
        )
        self.producer_thread.start()

    def _producer(self):
        """Background thread: push packets to queue at configured rate."""
        delay_per_sample = (1.0 / self.fs) / self.speed if self.speed > 0 else 0

        try:
            for i in range(self.n_samples):
                if not self._running:
                    break

                packet = self.data[i]  # Shape: (n_channels,)

                # Block if queue is full (backpressure)
                self.packet_queue.put(packet, timeout=1.0)

                if delay_per_sample > 0:
                    time.sleep(delay_per_sample)

        except Exception as e:
            print(f"Simulated source producer error: {e}")
        finally:
            self._thread_finished = True
            print(f"Simulated source finished ({self.n_samples:,} samples sent)")

    def get_packet(self) -> np.ndarray | None:
        """Get next packet from queue.

        Returns:
            np.ndarray of shape (n_channels,) or None if queue is empty
        """
        if not self._running:
            return None

        try:
            # Non-blocking get with short timeout
            packet = self.packet_queue.get(timeout=0.1)
            return np.array(packet, dtype=np.float64)
        except queue.Empty:
            return None

    def is_running(self) -> bool:
        """Check if more data is coming.

        Returns:
            False when producer thread has finished and queue is empty
        """
        return self._running and (not self._thread_finished or not self.packet_queue.empty())

    def close(self) -> None:
        """Stop the producer thread and clean up."""
        self._running = False
        if self.producer_thread.is_alive():
            self.producer_thread.join(timeout=2.0)
        print("Simulated source closed")
