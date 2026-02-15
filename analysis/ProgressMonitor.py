import numpy as np


class ProgressMonitor:
    """Tracks session progress and detects coordination plateaus or
    low-uniqueness stagnation in movement attempts.

    Called by MainPipeline after every scored activation. Returns a
    status dict that can trigger LLM interventions.
    """

    def __init__(
        self,
        plateau_window: int = 10,
        plateau_threshold: float = 0.05,
        uniqueness_threshold: float = 0.15,
    ):
        """
        Args:
            plateau_window: Number of recent attempts to consider when
                detecting a coordination plateau.
            plateau_threshold: Maximum standard deviation of the last
                ``plateau_window`` coordination indices before the
                session is considered plateaued.
            uniqueness_threshold: Maximum mean similarity score (DTW
                distance) among recent attempts before the session is
                flagged as low-uniqueness (patient repeating the same
                motion).
        """
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.uniqueness_threshold = uniqueness_threshold

        # Rolling logs
        self.coordination_history: list[float] = []
        self.similarity_history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_attempt(
        self,
        coordination_index: float,
        similarity_score: float,
    ) -> dict:
        """Log an attempt and run stagnation detection.

        Returns:
            dict with keys:
                plateau_detected (bool)
                plateau_type (str | None): "coordination_plateau",
                    "low_uniqueness", or None
                details (str): human-readable summary for the LLM
                trend (str): "improving", "declining", or "plateau"
        """
        self.coordination_history.append(coordination_index)
        self.similarity_history.append(similarity_score)

        trend = self.get_trend()

        # Check for plateau types (need enough data first)
        coord_plateau = self._detect_coordination_plateau()
        low_unique = self._detect_low_uniqueness()

        if coord_plateau:
            recent = self.coordination_history[-self.plateau_window:]
            std = float(np.std(recent))
            mean = float(np.mean(recent))
            return {
                "plateau_detected": True,
                "plateau_type": "coordination_plateau",
                "details": (
                    f"Over the last {self.plateau_window} attempts the "
                    f"coordination index has barely changed (mean={mean:.4f}, "
                    f"std={std:.4f}). The patient appears stuck."
                ),
                "trend": trend,
            }

        if low_unique:
            recent_sim = self.similarity_history[-self.plateau_window:]
            mean_sim = float(np.mean(recent_sim))
            return {
                "plateau_detected": True,
                "plateau_type": "low_uniqueness",
                "details": (
                    f"The last {self.plateau_window} attempts are very similar "
                    f"to each other (mean DTW distance={mean_sim:.4f}). The "
                    "patient is repeating the same motion pattern without "
                    "exploring new strategies."
                ),
                "trend": trend,
            }

        return {
            "plateau_detected": False,
            "plateau_type": None,
            "details": "",
            "trend": trend,
        }

    def get_trend(self, window: int = 10) -> str:
        """Return 'improving', 'declining', or 'plateau' based on a simple
        linear regression slope over the last ``window`` coordination indices.

        Note: for coordination index, *lower* is better, so a negative
        slope means the patient is improving.
        """
        n = len(self.coordination_history)
        if n < 3:
            return "plateau"  # not enough data to determine trend

        recent = self.coordination_history[-window:]
        m = len(recent)
        x = np.arange(m, dtype=float)
        y = np.array(recent, dtype=float)

        # Simple OLS slope: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator == 0:
            return "plateau"

        slope = numerator / denominator

        # Threshold the slope (negative slope = improving since lower CI is better)
        if slope < -0.005:
            return "improving"
        elif slope > 0.005:
            return "declining"
        else:
            return "plateau"

    def reset(self) -> None:
        """Clear all history (e.g. when switching exercises)."""
        self.coordination_history.clear()
        self.similarity_history.clear()

    # ------------------------------------------------------------------
    # Internal detection methods
    # ------------------------------------------------------------------

    def _detect_coordination_plateau(self) -> bool:
        """Check if the coordination index has flatlined over the
        recent window."""
        if len(self.coordination_history) < self.plateau_window:
            return False

        recent = self.coordination_history[-self.plateau_window:]
        std = float(np.std(recent))
        return std < self.plateau_threshold

    def _detect_low_uniqueness(self) -> bool:
        """Check if recent attempts are all very similar to each other
        (low DTW distances = same motion repeated)."""
        if len(self.similarity_history) < self.plateau_window:
            return False

        recent = self.similarity_history[-self.plateau_window:]
        mean_sim = float(np.mean(recent))
        return mean_sim < self.uniqueness_threshold
