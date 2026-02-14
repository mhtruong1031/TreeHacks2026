import numpy as np
from scipy.signal import decimate, resample


class PreprocessingPipeline:
    """Preprocessing pipeline that downsamples windowed time/voltage data to 100 Hz."""

    def __init__(self, target_fs: float = 100.0):
        """
        Args:
            target_fs: Desired output sampling rate in Hz (default 100 Hz).
        """
        self.target_fs = target_fs

    def _estimate_fs(self, times: np.ndarray) -> float:
        """Estimate the sampling frequency from a time vector.

        Args:
            times: 1-D array of timestamps (seconds).

        Returns:
            Estimated sampling rate in Hz.
        """
        dt = np.median(np.diff(times))
        if dt <= 0:
            raise ValueError("Time values must be monotonically increasing.")
        return 1.0 / dt

    def downsample_window(
        self,
        times: np.ndarray,
        voltages: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Downsample a single window of time/voltage data to 100 Hz.

        Takes one window of raw samples, estimates the original sampling rate,
        and returns a new (time, voltage) pair downsampled to ``self.target_fs``.

        Args:
            times:    1-D array of timestamps (seconds) for this window.
            voltages: 1-D array of voltage values for this window.

        Returns:
            (ds_times, ds_voltages) — downsampled NumPy arrays at ``target_fs``.
        """
        times = np.asarray(times, dtype=np.float64)
        voltages = np.asarray(voltages, dtype=np.float64)

        if len(times) != len(voltages):
            raise ValueError(
                f"times and voltages must have the same length, "
                f"got {len(times)} and {len(voltages)}."
            )
        if len(times) < 2:
            raise ValueError("Window must contain at least 2 samples.")

        original_fs = self._estimate_fs(times)

        # Already at or below target — return as-is
        if original_fs <= self.target_fs:
            return times.copy(), voltages.copy()

        ratio = original_fs / self.target_fs

        if ratio == int(ratio) and int(ratio) >= 2:
            # Integer decimation (anti-alias filter built in)
            q = int(ratio)
            ds_voltages = decimate(voltages, q, zero_phase=True)
            ds_times = times[::q][: len(ds_voltages)]
        else:
            # Arbitrary ratio — resample to nearest output length
            n_out = max(1, int(round(len(voltages) * self.target_fs / original_fs)))
            ds_voltages = resample(voltages, n_out)
            ds_times = np.linspace(times[0], times[-1], n_out)

        return ds_times, ds_voltages
