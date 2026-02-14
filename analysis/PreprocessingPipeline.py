import numpy as np
from scipy.signal import decimate, resample


class PreprocessingPipeline:
    """Pipeline to intake live time/voltage data and downsample to a target rate (default 100 Hz)."""

    def __init__(self, target_fs: float = 100.0):
        """
        Args:
            target_fs: Desired output sampling rate in Hz (default 100 Hz).
        """
        self.target_fs = target_fs
        # Rolling buffers for streaming intake
        self._time_buffer: list[float] = []
        self._voltage_buffer: list[float] = []

    # ------------------------------------------------------------------ #
    #  Data intake
    # ------------------------------------------------------------------ #

    def add_sample(self, t: float, v: float) -> None:
        """Append a single (time, voltage) sample to the internal buffer."""
        self._time_buffer.append(t)
        self._voltage_buffer.append(v)

    def add_samples(self, times, voltages) -> None:
        """Append a batch of samples (array-like) to the internal buffer."""
        self._time_buffer.extend(times)
        self._voltage_buffer.extend(voltages)

    def set_data(self, times, voltages) -> None:
        """Replace the internal buffer with a complete dataset."""
        self._time_buffer = list(times)
        self._voltage_buffer = list(voltages)

    # ------------------------------------------------------------------ #
    #  Sampling-rate estimation
    # ------------------------------------------------------------------ #

    def estimate_fs(self, times=None) -> float:
        """Estimate the original sampling frequency from the time vector.

        Args:
            times: Optional explicit time array. Falls back to the internal buffer.

        Returns:
            Estimated sampling rate in Hz.
        """
        t = np.asarray(times if times is not None else self._time_buffer)
        if len(t) < 2:
            raise ValueError("Need at least 2 samples to estimate sampling rate.")
        dt = np.median(np.diff(t))
        if dt <= 0:
            raise ValueError("Time values must be monotonically increasing.")
        return 1.0 / dt

    # ------------------------------------------------------------------ #
    #  Downsampling
    # ------------------------------------------------------------------ #

    def downsample(self, times=None, voltages=None):
        """Downsample time/voltage arrays to ``self.target_fs``.

        Uses ``scipy.signal.decimate`` when the ratio is an integer,
        otherwise falls back to ``scipy.signal.resample`` for arbitrary
        rate conversion.

        Args:
            times:    Optional time array (defaults to internal buffer).
            voltages: Optional voltage array (defaults to internal buffer).

        Returns:
            (ds_times, ds_voltages) — NumPy arrays at the target rate.
        """
        t = np.asarray(times if times is not None else self._time_buffer, dtype=np.float64)
        v = np.asarray(voltages if voltages is not None else self._voltage_buffer, dtype=np.float64)

        if len(t) < 2:
            raise ValueError("Need at least 2 samples to downsample.")

        original_fs = self.estimate_fs(t)

        if original_fs <= self.target_fs:
            # Already at or below target rate — nothing to do
            return t.copy(), v.copy()

        ratio = original_fs / self.target_fs

        if ratio == int(ratio) and int(ratio) >= 2:
            # Integer decimation factor — use decimate (anti-alias filter built-in)
            q = int(ratio)
            ds_voltage = decimate(v, q, zero_phase=True)
            ds_time = t[::q][: len(ds_voltage)]
        else:
            # Arbitrary ratio — resample to the nearest number of output samples
            n_out = int(round(len(v) * self.target_fs / original_fs))
            ds_voltage = resample(v, n_out)
            ds_time = np.linspace(t[0], t[-1], n_out)

        return ds_time, ds_voltage