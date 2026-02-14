import numpy as np
from scipy.signal import butter, filtfilt, decimate, resample


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

    def lowpass_blink_filter(
        self,
        signal: np.ndarray,
        fs: float,
        cutoff: float = 30.0,
        order: int = 4,
    ) -> np.ndarray:
        """Low-pass Butterworth filter to attenuate sharp blink spike transients.

        Eye blinks produce large, fast negative deflections whose sharp edges
        contain high-frequency energy.  A low-pass filter smooths out those
        transients while preserving the slower brain-relevant frequencies.

        Args:
            signal:  1-D voltage array (single channel).
            fs:      Sampling frequency in Hz.
            cutoff:  Low-pass cutoff frequency in Hz (default 30 Hz).
                     Frequencies *above* this are attenuated.
            order:   Butterworth filter order (default 4).

        Returns:
            Filtered 1-D voltage array (same length as input).
        """
        nyquist = fs / 2.0
        if cutoff >= nyquist:
            raise ValueError(
                f"Cutoff ({cutoff} Hz) must be below the Nyquist frequency ({nyquist} Hz)."
            )
        b, a = butter(order, cutoff / nyquist, btype="low")
        return filtfilt(b, a, signal)

    def bandstop_sweat_filter(
        self,
        signal: np.ndarray,
        fs: float,
        center: float = 0.5,
        width: float = 0.4,
        order: int = 4,
    ) -> np.ndarray:
        """Band-stop (notch) filter to remove sweat-related artifacts around 0.5 Hz.

        Sweat artifacts produce slow galvanic skin-potential drifts that
        concentrate around ~0.5 Hz.  This filter removes a narrow band
        centred on that frequency while preserving content above and below.

        Args:
            signal:  1-D voltage array (single channel).
            fs:      Sampling frequency in Hz.
            center:  Centre frequency of the stop band in Hz (default 0.5 Hz).
            width:   Full width of the stop band in Hz (default 0.4 Hz),
                     so the band spans [center - width/2, center + width/2],
                     i.e. [0.3, 0.7] Hz by default.
            order:   Butterworth filter order (default 4).

        Returns:
            Filtered 1-D voltage array (same length as input).
        """
        nyquist = fs / 2.0
        low = (center - width / 2.0) / nyquist
        high = (center + width / 2.0) / nyquist

        if low <= 0:
            raise ValueError(
                f"Lower stop-band edge ({center - width / 2.0} Hz) must be > 0 Hz."
            )
        if high >= 1.0:
            raise ValueError(
                f"Upper stop-band edge ({center + width / 2.0} Hz) must be "
                f"below the Nyquist frequency ({nyquist} Hz)."
            )

        b, a = butter(order, [low, high], btype="bandstop")
        return filtfilt(b, a, signal)

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
