import numpy as np
from scipy.signal import decimate, resample


class PreprocessingPipeline:
    """Preprocessing pipeline that downsamples a multi-channel time x voltage matrix to 100 Hz."""

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

    def _downsample_single_channel(
        self,
        signal: np.ndarray,
        original_fs: float,
    ) -> np.ndarray:
        """Downsample a single 1-D signal from original_fs to target_fs.

        Args:
            signal:      1-D voltage array for one channel.
            original_fs: Original sampling rate in Hz.

        Returns:
            Downsampled 1-D voltage array.
        """
        ratio = original_fs / self.target_fs

        if ratio == int(ratio) and int(ratio) >= 2:
            q = int(ratio)
            return decimate(signal, q, zero_phase=True)
        else:
            n_out = max(1, int(round(len(signal) * self.target_fs / original_fs)))
            return resample(signal, n_out)

    def downsample_window(self, data: np.ndarray) -> np.ndarray:
        """Downsample a multi-channel window to ``target_fs`` (default 100 Hz).

        Input and output have the **same shape convention**:
            rows    = time samples
            columns = channels  (column 0 is time, columns 1..N are voltage channels)

        The returned array has the same number of columns and the same
        column meaning; only the number of rows (time samples) is reduced.

        Args:
            data: 2-D array of shape (n_samples, 1 + n_channels).
                  Column 0  — timestamps (seconds).
                  Columns 1…N — voltage channels.

        Returns:
            Downsampled 2-D array with the same column layout
            (n_downsampled_samples, 1 + n_channels).
        """
        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                "data must be a 2-D array with at least 2 columns "
                "(time + at least one voltage channel)."
            )
        if data.shape[0] < 2:
            raise ValueError("Window must contain at least 2 time samples (rows).")

        times = data[:, 0]
        original_fs = self._estimate_fs(times)

        # Already at or below target — return a copy with identical shape
        if original_fs <= self.target_fs:
            return data.copy()

        # Downsample each voltage channel independently
        n_channels = data.shape[1] - 1
        ds_channels = [
            self._downsample_single_channel(data[:, ch + 1], original_fs)
            for ch in range(n_channels)
        ]

        # Build downsampled time column to match the new length
        n_out = len(ds_channels[0])
        ratio = original_fs / self.target_fs

        if ratio == int(ratio) and int(ratio) >= 2:
            ds_times = times[:: int(ratio)][:n_out]
        else:
            ds_times = np.linspace(times[0], times[-1], n_out)

        # Reassemble into the same (n_samples, 1 + n_channels) layout
        out = np.column_stack([ds_times] + ds_channels)
        return out
