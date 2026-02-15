"""
UDP data source for live OpenBCI data streams.

Wraps a UDP socket and reuses the parse_packet() logic from udp_receiver.py.
"""

import socket
import numpy as np
from .base import DataSource


def parse_packet(raw: bytes):
    """Parse UDP packet into samples (reused from udp_receiver.py).

    Returns:
        (packet_type, samples) where samples is list of lists [[ch0, ch1, ...], ...]
        or (None, None) if parsing failed.
    """
    import json

    def _flatten_or_wrap(lst):
        """Convert to list of samples."""
        if not lst:
            return []
        if isinstance(lst[0], list):
            return [[float(v) for v in row] for row in lst]
        return [[float(v) for v in lst]]

    text = raw.decode("utf-8", errors="replace").strip()

    # JSON parsing
    if text.startswith("{") or text.startswith("["):
        try:
            obj = json.loads(text)

            if isinstance(obj, dict):
                pkt_type = obj.get("type", None)
                for key in ("data", "channel_data", "eeg", "values", "samples"):
                    if key in obj and isinstance(obj[key], list):
                        return pkt_type, _flatten_or_wrap(obj[key])
                for v in obj.values():
                    if isinstance(v, list):
                        return pkt_type, _flatten_or_wrap(v)

            if isinstance(obj, list):
                return None, _flatten_or_wrap(obj)

        except (json.JSONDecodeError, ValueError):
            pass

    # CSV / whitespace-separated
    for sep in (",", "\t", " "):
        parts = text.split(sep)
        if len(parts) >= 2:
            try:
                row = [float(p) for p in parts if p.strip()]
                return None, [row]
            except ValueError:
                continue

    # Single value
    try:
        return None, [[float(text)]]
    except ValueError:
        return None, None


class UDPDataSource(DataSource):
    """Live UDP data source for OpenBCI streams."""

    def __init__(self, ip: str = "0.0.0.0", port: int = 12345, timeout: float = 1.0):
        """
        Args:
            ip: IP address to bind to (default: 0.0.0.0 = all interfaces)
            port: UDP port to listen on (default: 12345)
            timeout: Socket timeout in seconds (default: 1.0)
        """
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self._running = True
        self._first_packet = True

        # Create and bind socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((ip, port))
        self.sock.settimeout(timeout)

        print(f"UDP source listening on {ip}:{port}")

    def get_packet(self) -> np.ndarray | None:
        """Receive and parse next UDP packet.

        Returns:
            np.ndarray of shape (n_channels,) or None on timeout/error
        """
        if not self._running:
            return None

        try:
            data, addr = self.sock.recvfrom(65535)

            if self._first_packet:
                print(f"First packet received from {addr[0]}:{addr[1]}")
                self._first_packet = False

            pkt_type, samples = parse_packet(data)

            if samples is None or len(samples) == 0:
                return None

            # Return first sample as 1-D array
            return np.array(samples[0], dtype=np.float64)

        except socket.timeout:
            return None
        except Exception as e:
            print(f"UDP receive error: {e}")
            return None

    def is_running(self) -> bool:
        """UDP source runs indefinitely until closed."""
        return self._running

    def close(self) -> None:
        """Close the UDP socket."""
        self._running = False
        self.sock.close()
        print("UDP source closed")
