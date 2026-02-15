"""
Abstract base class for data sources.

All data sources (UDP, simulated, etc.) implement this interface
to provide a unified way to retrieve packets.
"""

from abc import ABC, abstractmethod
import numpy as np


class DataSource(ABC):
    """Abstract interface for EEG/EMG data sources."""

    @abstractmethod
    def get_packet(self) -> np.ndarray | None:
        """Get the next data packet.

        Returns:
            np.ndarray of shape (n_channels,) for a single sample,
            or None if no data is available (timeout, end of stream, error).
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the data source is still active.

        Returns:
            True if more data may arrive, False if stream has ended.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources (sockets, threads, files)."""
        pass
