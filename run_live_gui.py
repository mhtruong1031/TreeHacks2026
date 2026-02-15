#!/usr/bin/env python3
"""
Integration script to run LiveGUI with the existing pipeline.

Usage:
    # With simulated data
    python run_live_gui.py --simulate test_data.csv --speed 1.0

    # With live UDP data
    python run_live_gui.py --live --ip 0.0.0.0 --port 12345

    # Skip calibration
    python run_live_gui.py --simulate test_data.csv --skip-calibration
"""

import sys
import argparse
import threading
import time
import numpy as np
from PyQt5.QtWidgets import QApplication

from gui import LiveGUI
from data_sources import UDPDataSource, SimulatedDataSource
from analysis.MainPipeline import MainPipeline
from analysis.BaselineCalibration import BaselineCalibration


class GUIController:
    """Controller to connect data pipeline to GUI."""

    def __init__(self, gui, data_source, calibration_time_s=60.0, skip_calibration=False):
        self.gui = gui
        self.data_source = data_source
        self.calibration_time_s = calibration_time_s
        self.skip_calibration = skip_calibration
        self.pipeline = None
        self.calibration = None
        self.running = True

    def run_calibration(self):
        """Run calibration phase."""
        if self.skip_calibration:
            print("Skipping calibration...")
            return

        print(f"Starting calibration ({self.calibration_time_s}s)...")
        n_samples_needed = int(self.calibration_time_s * 200.0)
        rest_data = []

        while len(rest_data) < n_samples_needed and self.data_source.is_running():
            packet = self.data_source.get_packet()
            if packet is not None:
                # Use only first 4 channels (EMG + 3 EEG); ignore extra channels
                rest_data.append(np.asarray(packet).reshape(-1)[:4])
            else:
                time.sleep(0.001)

        if len(rest_data) >= n_samples_needed:
            rest_data_array = np.array(rest_data)
            self.calibration = BaselineCalibration(fs=200.0)
            self.calibration.calibrate(rest_data_array)
            print("Calibration complete!")

    def run_streaming(self):
        """Run main data processing loop."""
        print("Starting data streaming...")

        # Initialize pipeline
        self.pipeline = MainPipeline(
            window_size_s=0.2,
            activation_threshold=0.2,
            calibration=self.calibration
        )

        # Set up callbacks for immediate GUI updates when async operations complete
        self.pipeline.on_prediction_ready = self._on_prediction_ready
        self.pipeline.on_similarity_ready = self._on_similarity_ready

        packet_count = 0
        batch = []
        coord_update_counter = 0

        while self.running and self.data_source.is_running():
            packet = self.data_source.get_packet()

            if packet is None:
                time.sleep(0.001)
                continue

            # Use only first 4 channels (EMG + 3 EEG); ignore extra channels
            packet = np.asarray(packet).reshape(-1)[:4]

            # Process through pipeline
            self.pipeline.run(packet)
            packet_count += 1

            # Accumulate processed data
            if len(self.pipeline.processed_data) > 0:
                # Get latest processed sample
                latest = self.pipeline.processed_data[-1]
                batch.append(latest)

                # Send batch to GUI every 10 samples (~50ms @ 200Hz)
                if len(batch) >= 10:
                    self.gui.add_data(np.array(batch))
                    batch = []

                # Update coordination attempts display every 100 packets (~0.5s @ 200Hz)
                coord_update_counter += 1
                if coord_update_counter >= 100:
                    self._update_coordination_display()
                    coord_update_counter = 0

        print("Streaming stopped.")

    def _on_prediction_ready(self):
        """Callback when prediction completes - immediately update GUI."""
        print("[DEBUG] Prediction ready - triggering immediate GUI update")
        self._update_coordination_display()

    def _on_similarity_ready(self):
        """Callback when similarity scores update - immediately update GUI."""
        print("[DEBUG] Similarity scores ready - triggering immediate GUI update")
        self._update_coordination_display()

    def _update_coordination_display(self):
        """Extract top attempts from pipeline and update GUI."""
        if not hasattr(self.pipeline, 'max_n_coord_cache'):
            print("[DEBUG] Pipeline doesn't have max_n_coord_cache yet")
            return

        try:
            # Lock the cache while reading
            with self.pipeline.cache_lock:
                top_nodes = self.pipeline.max_n_coord_cache.get_top_n_nodes(5)
                predicted_ideal_node = self.pipeline.max_n_coord_cache.predicted_ideal

            print(f"[DEBUG] Retrieved {len(top_nodes)} top nodes from cache")

            # Convert top attempts to dict format for GUI
            top_attempts = []
            for i, node in enumerate(top_nodes, 1):
                print(f"[DEBUG] Node {i}: coord={node.coordination_index:.4f}, sim={node.similarity_score}")
                top_attempts.append({
                    'coordination_index': node.coordination_index,
                    'similarity_score': node.similarity_score if node.similarity_score is not None else 0.0,
                    'attempt_id': i  # Use position as ID for now
                })

            # Convert predicted ideal to dict format
            predicted_ideal = None
            if predicted_ideal_node is not None:
                print(f"[DEBUG] Predicted ideal: coord={predicted_ideal_node.coordination_index:.4f}, sim={predicted_ideal_node.similarity_score}")
                predicted_ideal = {
                    'coordination_index': predicted_ideal_node.coordination_index,
                    'similarity_score': predicted_ideal_node.similarity_score if predicted_ideal_node.similarity_score is not None else 0.0
                }
            else:
                print("[DEBUG] No predicted ideal node yet")

            # Update GUI
            if len(top_attempts) > 0:
                self.gui.update_coordination_attempts(top_attempts, predicted_ideal)
            else:
                print("[DEBUG] No attempts to display yet")

        except Exception as e:
            # Print errors instead of silently ignoring
            print(f"[ERROR] Failed to update coordination display: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Run calibration then streaming in background thread."""
        def background_task():
            try:
                self.run_calibration()
                self.run_streaming()
            except Exception as e:
                print(f"Error in background task: {e}")
                import traceback
                traceback.print_exc()

        thread = threading.Thread(target=background_task, daemon=True)
        thread.start()

    def stop(self):
        """Stop the controller."""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Live GUI for EEG/EMG visualization")

    # Data source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--live", action="store_true", help="Use live UDP")
    source_group.add_argument("--simulate", type=str, help="CSV file to simulate")

    # Options
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed (default: 1.0)")
    parser.add_argument("--calibration-time", type=float, default=60.0, help="Calibration duration (s)")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="UDP IP")
    parser.add_argument("--port", type=int, default=12345, help="UDP port")
    parser.add_argument("--fs", type=float, default=200.0, help="Sampling rate")

    args = parser.parse_args()

    # Create data source
    if args.live:
        data_source = UDPDataSource(ip=args.ip, port=args.port, timeout=1.0)
    else:
        if args.speed == 0:
            print("Warning: Using speed=1.0 instead of 0 for GUI stability")
            args.speed = 1.0
        data_source = SimulatedDataSource(
            csv_path=args.simulate,
            fs=args.fs,
            speed=args.speed
        )

    # Create Qt application
    app = QApplication(sys.argv)

    # Create GUI
    gui = LiveGUI(
        window_size_s=10.0,
        sampling_rate=200.0,
        update_rate_hz=60
    )
    gui.show()

    # Create controller
    controller = GUIController(
        gui=gui,
        data_source=data_source,
        calibration_time_s=args.calibration_time,
        skip_calibration=args.skip_calibration
    )

    # Start background processing
    controller.run()

    # Handle close event
    def on_close():
        controller.stop()
        data_source.close()

    gui.closeEvent = lambda event: (on_close(), event.accept())

    # Run Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
