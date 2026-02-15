#!/usr/bin/env python3
"""
Quick test of LiveGUI with simulated data.
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from gui import LiveGUI
import numpy as np

app = QApplication(sys.argv)

# Create GUI
gui = LiveGUI(window_size_s=10.0, sampling_rate=200.0, update_rate_hz=60)
gui.show()

# Simulate data generation at 200 Hz
packet_count = 0

def generate_data():
    """Generate simulated EEG/EMG data."""
    global packet_count

    # Generate 10 samples (50ms worth at 200 Hz)
    n = 10
    t = packet_count + np.arange(n)

    # Simulate different frequencies for different channels
    data = np.column_stack([
        50 * np.sin(2 * np.pi * 10 * t / 200) + np.random.randn(n) * 5,  # EEG 1: 10 Hz
        50 * np.sin(2 * np.pi * 12 * t / 200) + np.random.randn(n) * 5,  # EEG 2: 12 Hz
        50 * np.sin(2 * np.pi * 8 * t / 200) + np.random.randn(n) * 5,   # EEG 3: 8 Hz
        100 * np.sin(2 * np.pi * 20 * t / 200) + np.random.randn(n) * 10, # EMG: 20 Hz, higher amplitude
    ])

    gui.add_data(data)
    packet_count += n

# Timer to simulate data at 200 Hz (batch of 10 every 50ms)
data_timer = QTimer()
data_timer.timeout.connect(generate_data)
data_timer.start(50)  # 50ms = 20 Hz batch rate, but 10 samples per batch = 200 Hz

# Simulate coordination attempts after 2 seconds
def show_coordination():
    print("Updating coordination circles...")
    # Attempts in random order - will be sorted by GUI (best first)
    attempts = [
        {'coordination_index': 0.2341, 'similarity_score': 0.48, 'attempt_id': 27},  # Fair
        {'coordination_index': 0.5124, 'similarity_score': 0.03, 'attempt_id': 8},   # Very poor
        {'coordination_index': 0.0845, 'similarity_score': 0.92, 'attempt_id': 52},  # Best!
        {'coordination_index': 0.3892, 'similarity_score': 0.19, 'attempt_id': 15},  # Poor
        {'coordination_index': 0.1523, 'similarity_score': 0.71, 'attempt_id': 41},  # Good
    ]
    # Predicted ideal - no coordination index shown, no attempt ID
    predicted_ideal = {
        'coordination_index': 0.0723,  # Not displayed
        'similarity_score': 1.0  # Perfect match
    }
    gui.update_coordination_attempts(attempts, predicted_ideal)

QTimer.singleShot(2000, show_coordination)

# Auto-close after 10 seconds for testing
QTimer.singleShot(10000, app.quit)

print("Starting GUI test (will run for 10 seconds)...")
print("You should see:")
print("  - Blue: EEG Channel 1 (10 Hz)")
print("  - Green: EEG Channel 2 (12 Hz)")
print("  - Cyan: EEG Channel 3 (8 Hz)")
print("  - Red: EMG (20 Hz, larger amplitude)")
print("  - After 2s: 5 circles showing top attempts + 1 predicted ideal circle")
print("    (Greener = higher similarity, darker = lower similarity)")
print("    (6th circle should be brightest green - perfect similarity)")
print()

sys.exit(app.exec_())
