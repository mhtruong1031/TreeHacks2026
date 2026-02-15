"""
Data source abstraction for unified UDP and simulated EEG/EMG data streams.

Provides a common interface for MainPipeline to receive packets from:
  - Live UDP (OpenBCI GUI)
  - CSV file playback (simulation/testing)
"""

from .base import DataSource
from .udp_source import UDPDataSource
from .simulated_source import SimulatedDataSource

__all__ = ['DataSource', 'UDPDataSource', 'SimulatedDataSource']
