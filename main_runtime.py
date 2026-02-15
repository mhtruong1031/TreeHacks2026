"""
Unified runtime system for EEG/EMG signal processing with baseline calibration.

Supports two-phase execution:
  1. Calibration phase: Collect 60s rest data → compute baseline statistics
  2. Streaming phase: Process live/simulated data with adaptive thresholds

Usage examples:
    # Live UDP with calibration
    python main_runtime.py --live --calibration-time 60

    # Simulated unlimited speed
    python main_runtime.py --simulate trial_001.csv --speed 0

    # With whitening
    python main_runtime.py --simulate trial.csv --whitening

    # Skip calibration (legacy mode)
    python main_runtime.py --simulate test.csv --skip-calibration
"""

import argparse
import time
import sys
import numpy as np

from data_sources import DataSource, UDPDataSource, SimulatedDataSource
from analysis.BaselineCalibration import BaselineCalibration
from analysis.MainPipeline import MainPipeline


class RuntimeOrchestrator:
    """Orchestrates two-phase execution: calibration → streaming."""

    def __init__(
        self,
        data_source: DataSource,
        calibration_time_s: float = 60.0,
        skip_calibration: bool = False,
        use_whitening: bool = False,
        adaptive_threshold_n_std: float = 2.5,
        fs: float = 200.0,
        top_n_attempts: int = 5
    ):
        """
        Args:
            data_source: DataSource instance (UDP or simulated)
            calibration_time_s: Duration of calibration phase in seconds (default: 60s)
            skip_calibration: If True, use fixed thresholds instead of calibration
            use_whitening: Enable spatial whitening after calibration
            adaptive_threshold_n_std: Std multiplier for adaptive thresholds (default: 2.5)
            fs: Sampling rate in Hz (default: 200.0)
            top_n_attempts: Number of top attempts to display in status (default: 5)
        """
        self.data_source = data_source
        self.calibration_time_s = calibration_time_s
        self.skip_calibration = skip_calibration
        self.use_whitening = use_whitening
        self.adaptive_threshold_n_std = adaptive_threshold_n_std
        self.fs = fs
        self.top_n_attempts = top_n_attempts

        self.calibration: BaselineCalibration | None = None
        self.pipeline: MainPipeline | None = None

    def run(self):
        """Execute the full runtime: calibration → streaming."""
        print("=" * 70)
        print("  EEG/EMG Real-Time Processing System")
        print("=" * 70)
        print()

        try:
            # Phase 1: Calibration
            if not self.skip_calibration:
                print("┌─────────────────────────────────────────────────────┐")
                print("│  PHASE 1: BASELINE CALIBRATION                      │")
                print("└─────────────────────────────────────────────────────┘")
                print()
                self._run_calibration_phase()
                print()
            else:
                print("⚠️  Calibration skipped — using fixed thresholds")
                print()

            # Phase 2: Streaming
            print("┌─────────────────────────────────────────────────────┐")
            print("│  PHASE 2: STREAMING ANALYSIS                        │")
            print("└─────────────────────────────────────────────────────┘")
            print()
            self._run_streaming_phase()

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        finally:
            self.data_source.close()
            print("\n✓ Runtime complete")

    def _run_calibration_phase(self):
        """Collect resting-state data and calibrate baseline statistics."""
        n_samples_needed = int(self.calibration_time_s * self.fs)
        rest_data = []

        print(f"Please sit still and relax for {self.calibration_time_s:.0f} seconds...")
        print(f"Collecting {n_samples_needed:,} samples @ {self.fs} Hz")
        print()

        t_start = time.perf_counter()
        packet_count = 0
        none_count = 0
        max_none = 100  # Allow up to 100 consecutive None packets before giving up
        last_report_time = t_start

        while len(rest_data) < n_samples_needed and self.data_source.is_running():
            packet = self.data_source.get_packet()

            if packet is None:
                none_count += 1
                if none_count >= max_none:
                    print(f"\n⚠️  No data received after {max_none} attempts")
                    print("Check that data source is sending packets")
                    sys.exit(1)
                time.sleep(0.01)  # Brief pause before retry
                continue

            none_count = 0  # Reset counter on successful packet
            rest_data.append(packet)
            packet_count += 1

            # Progress report every 1 second
            current_time = time.perf_counter()
            if current_time - last_report_time >= 1.0:
                elapsed = current_time - t_start
                progress = len(rest_data) / n_samples_needed * 100
                rate = packet_count / elapsed if elapsed > 0 else 0
                time_remaining = (n_samples_needed - len(rest_data)) / rate if rate > 0 else 0
                print(f"  [{progress:5.1f}%] {len(rest_data):>6,} / {n_samples_needed:,} samples  |  "
                      f"{rate:6.1f} Hz  |  {elapsed:5.1f}s  |  ~{time_remaining:4.1f}s remaining")
                last_report_time = current_time

        elapsed = time.perf_counter() - t_start
        actual_rate = len(rest_data) / elapsed if elapsed > 0 else 0

        print()
        print(f"✓ Calibration data collected: {len(rest_data):,} samples in {elapsed:.1f}s ({actual_rate:.1f} Hz)")
        print()

        # Run calibration
        rest_data_array = np.array(rest_data)
        self.calibration = BaselineCalibration(fs=self.fs)
        self.calibration.calibrate(rest_data_array)
        print()

    def _run_streaming_phase(self):
        """Initialize MainPipeline and process streaming data."""
        # Initialize pipeline with calibration (if available)
        self.pipeline = MainPipeline(
            window_size_s=0.2,
            activation_threshold=0.2,  # Fallback for non-calibrated mode
            prediction_data_threshold=30,
            calibration=self.calibration,
            use_whitening=self.use_whitening,
            adaptive_threshold_n_std=self.adaptive_threshold_n_std
        )

        # Print calibration info
        calib_info = self.pipeline.get_calibration_info()
        self._print_calibration_summary(calib_info)
        print()

        print("Processing packets... (Ctrl+C to stop)")
        print("-" * 70)
        print()

        t_start = time.perf_counter()
        packet_count = 0
        last_report_time = t_start
        last_report_packet_count = 0

        while self.data_source.is_running():
            packet = self.data_source.get_packet()

            if packet is None:
                time.sleep(0.001)  # Brief pause if no data
                continue

            # Process packet
            self.pipeline.run(packet)
            packet_count += 1

            # Status report every 1 second (time-based instead of packet-based)
            current_time = time.perf_counter()
            if current_time - last_report_time >= 1.0:
                elapsed = current_time - t_start
                interval = current_time - last_report_time
                packets_in_interval = packet_count - last_report_packet_count
                rate = packets_in_interval / interval if interval > 0 else 0

                self._print_status_report(packet_count, elapsed, rate)

                last_report_time = current_time
                last_report_packet_count = packet_count

        # Final report
        elapsed = time.perf_counter() - t_start
        print()
        print("=" * 70)
        print(f"✓ Streaming complete")
        print(f"  Total packets: {packet_count:,}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average rate: {packet_count / elapsed:.1f} Hz")
        print("=" * 70)

        # Final pipeline status
        self._print_final_summary()

    def _print_calibration_summary(self, info: dict):
        """Print calibration configuration."""
        print("Calibration Status:")
        if info['calibrated']:
            print(f"  Mode: ✓ Adaptive thresholds ({info['adaptive_threshold_n_std']:.1f} std)")
            print(f"  Channel means: {[f'{m:.4f}' for m in info['channel_means']]}")
            print(f"  Channel stds:  {[f'{s:.4f}' for s in info['channel_stds']]}")
            print(f"  Thresholds:    {[f'{t:.4f}' for t in info['adaptive_thresholds']]}")
            if info['whitening_enabled']:
                print(f"  Whitening: ✓ Enabled")
        else:
            print(f"  Mode: Fixed threshold = {info['fixed_threshold']}")

    def _print_status_report(self, packet_count: int, elapsed: float, rate: float):
        """Print periodic status update."""
        async_status = self.pipeline.get_async_status()
        calib_info = self.pipeline.get_calibration_info()

        # Attempt statistics
        n_attempts = async_status['num_attempts']
        model_status = "✓ Trained" if async_status['model_trained'] else f"Pending ({n_attempts}/30)"

        # Training status
        training_icon = "⏳" if async_status['training_in_progress'] else " "
        similarity_icon = "⏳" if async_status['similarity_in_progress'] else " "
        prediction_icon = "⏳" if async_status['prediction_in_progress'] else " "

        # Artifact rate
        artifact_info = ""
        if calib_info['calibrated']:
            artifact_info = f" | Artifacts: {calib_info['artifacts_rejected']} ({calib_info['artifact_rate']*100:.1f}%)"

        print(f"[{elapsed:7.1f}s] Packets: {packet_count:>7,} | Rate: {rate:6.1f} Hz | "
              f"Attempts: {n_attempts:>3} | Model: {model_status} | "
              f"T{training_icon} S{similarity_icon} P{prediction_icon}{artifact_info}")

        # Show top attempts if available (use configurable top_n)
        if n_attempts >= 3:
            self._print_top_attempts(n=min(self.top_n_attempts, n_attempts))

    def _print_top_attempts(self, n: int = 3):
        """Print top N attempts with coordination and similarity scores."""
        with self.pipeline.cache_lock:
            top_nodes = self.pipeline.max_n_coord_cache.get_top_n_nodes(n)
            predicted_ideal = self.pipeline.max_n_coord_cache.predicted_ideal

        if not top_nodes:
            return

        print("  Top attempts:")
        for i, node in enumerate(top_nodes, 1):
            coord = node.coordination_index
            sim = node.similarity_score if node.similarity_score is not None else 0.0
            print(f"    {i}. Coord: {coord:.4f} | Similarity: {sim:.4f}")

        # Show predicted ideal similarity if model is trained
        if predicted_ideal is not None and predicted_ideal.similarity_score is not None:
            print(f"  Similarity to predicted ideal: {predicted_ideal.similarity_score:.4f}")

    def _print_final_summary(self):
        """Print final pipeline statistics."""
        print()
        print("Final Statistics:")

        async_status = self.pipeline.get_async_status()
        calib_info = self.pipeline.get_calibration_info()

        print(f"  Total attempts cached: {async_status['num_attempts']}")
        print(f"  Model trained: {'Yes' if async_status['model_trained'] else 'No'}")

        if calib_info['calibrated']:
            print(f"  Artifacts rejected: {calib_info['artifacts_rejected']} "
                  f"({calib_info['artifact_rate']*100:.1f}%)")

        # Wait for async operations to complete
        print()
        print("Waiting for async operations to complete...")
        self.pipeline.wait_for_async_operations(timeout=30.0)
        print("✓ All operations complete")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time EEG/EMG processing with baseline calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live UDP with calibration
  python main_runtime.py --live --calibration-time 60

  # Simulated unlimited speed
  python main_runtime.py --simulate trial_001.csv --speed 0

  # With whitening
  python main_runtime.py --simulate trial.csv --whitening

  # Skip calibration (legacy mode)
  python main_runtime.py --simulate test.csv --skip-calibration
        """
    )

    # Data source selection (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--live",
        action="store_true",
        help="Use live UDP data source (OpenBCI)"
    )
    source_group.add_argument(
        "--simulate",
        type=str,
        metavar="CSV_PATH",
        help="Use simulated CSV data source"
    )

    # UDP options
    parser.add_argument(
        "--ip",
        type=str,
        default="0.0.0.0",
        help="UDP IP address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12345,
        help="UDP port (default: 12345)"
    )

    # Simulation options
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed for simulated data (0=unlimited, 1=real-time, 10=10x) (default: 1.0)"
    )

    # Calibration options
    parser.add_argument(
        "--calibration-time",
        type=float,
        default=60.0,
        help="Calibration duration in seconds (default: 60.0)"
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration and use fixed thresholds"
    )
    parser.add_argument(
        "--whitening",
        action="store_true",
        help="Enable spatial whitening (requires calibration)"
    )
    parser.add_argument(
        "--threshold-n-std",
        type=float,
        default=2.5,
        help="Adaptive threshold multiplier in std units (default: 2.5)"
    )

    # System options
    parser.add_argument(
        "--fs",
        type=float,
        default=200.0,
        help="Sampling rate in Hz (default: 200.0)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top attempts to display (default: 5)"
    )

    args = parser.parse_args()

    # Validate whitening option
    if args.whitening and args.skip_calibration:
        parser.error("--whitening requires calibration (don't use --skip-calibration)")

    # Create data source
    if args.live:
        data_source = UDPDataSource(ip=args.ip, port=args.port, timeout=1.0)
    else:
        data_source = SimulatedDataSource(
            csv_path=args.simulate,
            fs=args.fs,
            speed=args.speed
        )

    # Create and run orchestrator
    orchestrator = RuntimeOrchestrator(
        data_source=data_source,
        calibration_time_s=args.calibration_time,
        skip_calibration=args.skip_calibration,
        use_whitening=args.whitening,
        adaptive_threshold_n_std=args.threshold_n_std,
        fs=args.fs,
        top_n_attempts=args.top_n
    )

    orchestrator.run()


if __name__ == "__main__":
    main()
