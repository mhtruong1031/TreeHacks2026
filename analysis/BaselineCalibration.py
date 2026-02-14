"""
Baseline calibration from a resting-state recording.

During a ~60 s "just sit still" phase the class collects noise-only data
and derives per-subject parameters that are later applied to clean task data:

    1. Per-channel amplitude statistics  (mean, std)  → artifact thresholds
    2. Blink template                                  → template subtraction
    3. Noise power spectral density                    → spectral subtraction
    4. Noise covariance matrix                         → spatial whitening

Channel convention (after background subtraction):
    0 — muscle  (EMG)
    1 — left head (EEG)
    2 — right head (EEG)
"""

import numpy as np
from scipy.signal import welch, find_peaks
from scipy.linalg import inv, sqrtm


class BaselineCalibration:
    """Compute and store noise parameters from a resting baseline recording."""

    def __init__(self, fs: float = 200.0, blink_threshold_std: float = 4.0):
        """
        Args:
            fs:                  Sampling rate in Hz.
            blink_threshold_std: Number of standard deviations below the mean
                                 that counts as a blink in the brain channels.
        """
        self.fs = fs
        self.blink_threshold_std = blink_threshold_std

        # ── Outputs (populated after calibrate()) ────────────────────
        self.channel_mean: np.ndarray | None = None   # (n_channels,)
        self.channel_std: np.ndarray | None = None    # (n_channels,)
        self.blink_template: np.ndarray | None = None # (window_samples,)
        self.blink_pre_samples: int = 0
        self.blink_post_samples: int = 0
        self.noise_psd_freqs: np.ndarray | None = None
        self.noise_psd: np.ndarray | None = None      # (n_freq, n_channels)
        self.noise_cov: np.ndarray | None = None      # (n_channels, n_channels)
        self.whitening_matrix: np.ndarray | None = None

        self._calibrated = False

    # ─── Main entry point ────────────────────────────────────────────
    def calibrate(self, baseline_data: np.ndarray) -> None:
        """Run the full calibration from a resting-state recording.

        Args:
            baseline_data: 2-D array of shape (n_samples, n_channels).
                           Should already have background (Ch 0) subtracted,
                           so columns are [muscle, left_head, right_head].
        """
        baseline_data = np.asarray(baseline_data, dtype=np.float64)
        if baseline_data.ndim != 2 or baseline_data.shape[1] < 2:
            raise ValueError("baseline_data must be 2-D with at least 2 columns.")

        self._compute_statistics(baseline_data)
        self._extract_blink_template(baseline_data)
        self._compute_noise_psd(baseline_data)
        self._compute_noise_covariance(baseline_data)

        self._calibrated = True
        n_samples = baseline_data.shape[0]
        duration = n_samples / self.fs
        n_blinks = "none detected" if self.blink_template is None else "template ready"
        print(f"Calibration complete  ({duration:.1f}s, {n_samples:,} samples)")
        print(f"  Channel means: {self.channel_mean}")
        print(f"  Channel stds:  {self.channel_std}")
        print(f"  Blinks:        {n_blinks}")

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # ─── 1. Per-channel statistics ───────────────────────────────────
    def _compute_statistics(self, data: np.ndarray) -> None:
        self.channel_mean = np.mean(data, axis=0)
        self.channel_std = np.std(data, axis=0)

    # ─── 2. Blink template extraction ────────────────────────────────
    def _extract_blink_template(
        self,
        data: np.ndarray,
        pre_ms: float = 200.0,
        post_ms: float = 400.0,
    ) -> None:
        """Detect blinks in the brain channels and average into a template.

        Blinks are detected as large negative deflections that exceed
        ``blink_threshold_std`` standard deviations below the mean.
        A canonical template is built by averaging all detected blinks.

        Args:
            data:    (n_samples, n_channels) baseline array.
            pre_ms:  Milliseconds to include before each blink peak.
            post_ms: Milliseconds to include after each blink peak.
        """
        self.blink_pre_samples = int(pre_ms / 1000.0 * self.fs)
        self.blink_post_samples = int(post_ms / 1000.0 * self.fs)
        window_len = self.blink_pre_samples + self.blink_post_samples

        # Average the brain channels (assumed to be columns 1+)
        brain_channels = data[:, 1:]
        brain_avg = np.mean(brain_channels, axis=1)

        mean = np.mean(brain_avg)
        std = np.std(brain_avg)
        if std == 0:
            self.blink_template = None
            return

        threshold = mean - self.blink_threshold_std * std

        # Find negative peaks that cross the threshold
        inverted = -brain_avg  # invert so peaks become positive
        min_distance = int(0.3 * self.fs)  # blinks are at least 300 ms apart
        peak_indices, _ = find_peaks(
            inverted,
            height=-threshold,  # corresponds to original < threshold
            distance=min_distance,
        )

        # Extract windows around each peak
        templates = []
        for idx in peak_indices:
            start = idx - self.blink_pre_samples
            end = idx + self.blink_post_samples
            if start < 0 or end > len(brain_avg):
                continue
            templates.append(brain_avg[start:end])

        if len(templates) < 2:
            self.blink_template = None
            return

        self.blink_template = np.mean(templates, axis=0)  # (window_len,)
        print(f"  Blink template: {len(templates)} blinks detected, "
              f"window = {window_len} samples ({pre_ms + post_ms:.0f} ms)")

    # ─── 3. Noise power spectral density ─────────────────────────────
    def _compute_noise_psd(self, data: np.ndarray, nperseg: int = 1024) -> None:
        n_channels = data.shape[1]
        psds = []
        for ch in range(n_channels):
            f, psd = welch(data[:, ch], fs=self.fs, nperseg=min(nperseg, data.shape[0]))
            psds.append(psd)
        self.noise_psd_freqs = f
        self.noise_psd = np.column_stack(psds)  # (n_freq, n_channels)

    # ─── 4. Noise covariance & whitening matrix ──────────────────────
    def _compute_noise_covariance(self, data: np.ndarray) -> None:
        self.noise_cov = np.cov(data.T)  # (n_channels, n_channels)
        try:
            self.whitening_matrix = inv(sqrtm(self.noise_cov)).real
        except np.linalg.LinAlgError:
            self.whitening_matrix = None

    # ==================================================================
    #  Methods to APPLY calibration to task data
    # ==================================================================

    def subtract_blink_template(self, signal: np.ndarray) -> np.ndarray:
        """Detect and subtract blink artifacts from a 1-D brain channel signal.

        Uses the blink template extracted during calibration.  At each
        detected blink location the template is scaled (least-squares)
        and subtracted.

        Args:
            signal: 1-D voltage array (single brain channel).

        Returns:
            Cleaned 1-D array (same length).
        """
        self._check_calibrated()
        if self.blink_template is None:
            return signal.copy()

        cleaned = signal.copy()
        mean = self.channel_mean[1:].mean()  # brain channel mean
        std = self.channel_std[1:].mean()    # brain channel std
        threshold = mean - self.blink_threshold_std * std

        inverted = -cleaned
        min_distance = int(0.3 * self.fs)
        peak_indices, _ = find_peaks(
            inverted, height=-threshold, distance=min_distance
        )

        template = self.blink_template
        t_dot_t = np.dot(template, template)
        if t_dot_t == 0:
            return cleaned

        for idx in peak_indices:
            start = idx - self.blink_pre_samples
            end = idx + self.blink_post_samples
            if start < 0 or end > len(cleaned):
                continue
            segment = cleaned[start:end]
            scale = np.dot(segment, template) / t_dot_t
            cleaned[start:end] -= scale * template

        return cleaned

    def subtract_noise_psd(
        self, signal: np.ndarray, channel_idx: int
    ) -> np.ndarray:
        """Spectral subtraction: remove the baseline noise floor from a signal.

        Transforms to frequency domain, subtracts the calibrated noise PSD,
        clamps negatives to zero, and transforms back.

        Args:
            signal:      1-D voltage array.
            channel_idx: Column index (0=muscle, 1=left_head, 2=right_head)
                         to select the correct noise PSD.

        Returns:
            Cleaned 1-D array (same length).
        """
        self._check_calibrated()
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        spectrum = np.fft.rfft(signal)
        power = np.abs(spectrum) ** 2

        # Interpolate the noise PSD to match the FFT frequency grid
        noise_power = np.interp(
            freqs, self.noise_psd_freqs, self.noise_psd[:, channel_idx]
        )
        # Scale welch PSD estimate to match FFT power
        noise_power *= n

        clean_power = np.maximum(power - noise_power, 0)
        # Preserve phase, adjust magnitude
        magnitude_ratio = np.sqrt(clean_power / np.maximum(power, 1e-30))
        clean_spectrum = spectrum * magnitude_ratio

        return np.fft.irfft(clean_spectrum, n=n)

    def apply_whitening(self, data: np.ndarray) -> np.ndarray:
        """Apply spatial whitening using the baseline noise covariance.

        Decorrelates noise across channels so that downstream analysis
        (SVD, correlation) is not dominated by shared noise.

        Args:
            data: 2-D array of shape (n_samples, n_channels).

        Returns:
            Whitened 2-D array (same shape).
        """
        self._check_calibrated()
        if self.whitening_matrix is None:
            raise RuntimeError("Whitening matrix could not be computed.")
        return (self.whitening_matrix @ data.T).T

    def is_artifact(
        self, sample: np.ndarray, n_std: float = 5.0
    ) -> np.ndarray:
        """Flag samples that exceed baseline amplitude thresholds.

        Args:
            sample:  1-D array with one value per channel, or 2-D
                     (n_samples, n_channels).
            n_std:   Number of baseline standard deviations to use as
                     the rejection threshold.

        Returns:
            Boolean array — True where the sample exceeds the threshold.
        """
        self._check_calibrated()
        return np.abs(sample - self.channel_mean) > n_std * self.channel_std

    # ─── helpers ─────────────────────────────────────────────────────
    def _check_calibrated(self) -> None:
        if not self._calibrated:
            raise RuntimeError(
                "Calibration has not been run yet. Call calibrate() first."
            )
