"""
Live graphical interface for EEG/EMG data visualization.

Features:
- Real-time plotting at 60 Hz refresh rate
- 4-channel display (3 EEG + 1 EMG)
- 60% width × 60% height plot with margins
- Processes data at 200 Hz, displays at 60 Hz
"""

import sys
import numpy as np
from collections import deque
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QFrame)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QPainter, QColor, QPen
from PyQt5.QtCore import QRect

# Disable OpenGL for stability
pg.setConfigOption('useOpenGL', False)
pg.setConfigOption('antialias', False)


class CoordinationCircle(QWidget):
    """Widget to display a circle representing coordination attempt."""

    def __init__(self, parent=None, show_similarity=False, show_coordination=True):
        super().__init__(parent)
        self.setFixedSize(150, 155)  # Increased height for attempt number
        self.coordination_index = None
        self.similarity_score = 0.0
        self.attempt_id = None
        self.show_similarity = show_similarity
        self.show_coordination = show_coordination

    def set_data(self, coordination_index, similarity_score, attempt_id=None):
        """Update the circle's data."""
        self.coordination_index = coordination_index
        self.similarity_score = similarity_score
        self.attempt_id = attempt_id
        self.update()

    def paintEvent(self, event):
        """Draw the circle and text."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw attempt ID at very top if available
        top_offset = 0
        if self.attempt_id is not None:
            painter.setPen(QColor(0, 0, 0))
            attempt_font = QFont()
            attempt_font.setPointSize(8)
            attempt_font.setBold(True)
            painter.setFont(attempt_font)
            painter.drawText(QRect(0, 0, self.width(), 15), Qt.AlignCenter, f"Attempt {self.attempt_id}")
            top_offset = 15

        # Circle parameters
        circle_diameter = 60
        label_width = 70
        circle_x = label_width + 10
        circle_y = 15 + top_offset

        # Draw "Similarity" label
        painter.setPen(QColor(80, 80, 80))
        label_font = QFont()
        label_font.setPointSize(8)
        painter.setFont(label_font)
        sim_label_rect = QRect(0, circle_y + circle_diameter // 2 - 10, label_width, 20)
        painter.drawText(sim_label_rect, Qt.AlignRight | Qt.AlignVCenter, "Similarity:")

        if self.coordination_index is not None:
            # Calculate color based on similarity score (0 = black, 1 = bright green)
            green_intensity = int(255 * min(1, self.similarity_score*0.65))
            color = QColor(0, green_intensity, 0)

            # Draw filled circle
            painter.setBrush(color)
            painter.setPen(QPen(QColor(100, 100, 100), 2))  # Gray border
            painter.drawEllipse(circle_x, circle_y, circle_diameter, circle_diameter)

            # Draw similarity score on circle if enabled
            if self.show_similarity:
                painter.setPen(QColor(255, 255, 255))  # White text for visibility on green
                font = QFont()
                font.setPointSize(11)
                font.setBold(True)
                painter.setFont(font)
                sim_text = f"{self.similarity_score:.2f}"
                circle_rect = QRect(circle_x, circle_y, circle_diameter, circle_diameter)
                painter.drawText(circle_rect, Qt.AlignCenter, sim_text)

            # Draw "Coordination" label and value below circle (if enabled)
            if self.show_coordination:
                coord_y = circle_y + circle_diameter + 8

                # Draw "Coordination:" label on the left
                painter.setPen(QColor(80, 80, 80))  # Dark gray
                coord_label_font = QFont()
                coord_label_font.setPointSize(8)
                painter.setFont(coord_label_font)
                coord_label_rect = QRect(0, coord_y, label_width, 20)
                painter.drawText(coord_label_rect, Qt.AlignRight | Qt.AlignVCenter, "Coordination:")

                # Draw coordination index value on the right
                text = f"{self.coordination_index:.4f}"
                painter.setPen(QColor(0, 0, 0))  # Black text
                font = QFont()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                text_rect = QRect(label_width + 5, coord_y, self.width() - label_width - 5, 20)
                painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, text)
        else:
            # Draw empty circle (no data yet)
            painter.setBrush(QColor(50, 50, 50))  # Dark gray
            painter.setPen(QPen(QColor(100, 100, 100), 2))
            painter.drawEllipse(circle_x, circle_y, circle_diameter, circle_diameter)

            # Draw "Coordination:" label and placeholder (if enabled)
            if self.show_coordination:
                coord_y = circle_y + circle_diameter + 8

                # Draw label
                painter.setPen(QColor(80, 80, 80))
                coord_label_font = QFont()
                coord_label_font.setPointSize(8)
                painter.setFont(coord_label_font)
                coord_label_rect = QRect(0, coord_y, label_width, 20)
                painter.drawText(coord_label_rect, Qt.AlignRight | Qt.AlignVCenter, "Coordination:")

                # Draw placeholder text
                painter.setPen(QColor(100, 100, 100))  # Dark gray text
                font = QFont()
                font.setPointSize(10)
                painter.setFont(font)
                text_rect = QRect(label_width + 5, coord_y, self.width() - label_width - 5, 20)
                painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, "---")


class LiveGUI(QMainWindow):
    """
    Live graphical interface for real-time EEG/EMG visualization.

    Displays 4 channels:
    - Channels 0-2: EEG (Blue, Green, Cyan)
    - Channel 3: EMG (Red)
    """

    def __init__(self, window_size_s=10.0, sampling_rate=200.0, update_rate_hz=60):
        """
        Initialize the live GUI.

        Args:
            window_size_s: Time window to display (seconds)
            sampling_rate: Data sampling rate (Hz)
            update_rate_hz: Display refresh rate (Hz)
        """
        super().__init__()

        self.window_size_s = window_size_s
        self.sampling_rate = sampling_rate
        self.update_rate_hz = update_rate_hz
        self.buffer_size = int(window_size_s * sampling_rate)

        # Data buffers
        self.time_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.data_buffer = np.zeros((self.buffer_size, 4), dtype=np.float64)
        self.current_time = 0.0
        self.data_queue = deque(maxlen=1000)  # Queue for incoming data

        # Setup UI
        self._setup_window()
        self._setup_plot()
        self._setup_timer()

    def _setup_window(self):
        """Setup main window with margins."""
        self.setWindowTitle("EEG/EMG Live Monitor")

        # Window size
        window_width = 1280
        window_height = 720
        self.resize(window_width, window_height)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout with margins (20% total, 10% each side)
        main_layout = QVBoxLayout()
        margin_x = int(window_width * 0.1)
        margin_y = int(window_height * 0.1)
        main_layout.setContentsMargins(margin_x, margin_y, margin_x, margin_y)
        central.setLayout(main_layout)

        # Title
        title = QLabel("Real-Time EEG/EMG Monitor")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Plot container placeholder (will be replaced in _setup_plot)
        # GraphicsLayoutWidget allows stacked subplots
        self.plot_widget = pg.GraphicsLayoutWidget()
        plot_width = int(window_width * 0.6)
        plot_height = int(window_height * 0.6)
        self.plot_widget.setFixedSize(plot_width, plot_height)
        main_layout.addWidget(self.plot_widget, alignment=Qt.AlignCenter)

        # Coordination attempts visualization
        coord_frame = QFrame()
        coord_frame.setFrameShape(QFrame.StyledPanel)
        coord_layout = QVBoxLayout()
        coord_frame.setLayout(coord_layout)

        # Title for coordination section
        coord_title = QLabel("Top 5 Most Coordinated Attempts")
        coord_title_font = QFont()
        coord_title_font.setPointSize(12)
        coord_title_font.setBold(True)
        coord_title.setFont(coord_title_font)
        coord_title.setAlignment(Qt.AlignCenter)
        coord_layout.addWidget(coord_title)

        # Circles container - horizontal layout with divider
        circles_main_layout = QHBoxLayout()
        circles_main_layout.setSpacing(20)
        self.coord_circles = []

        # First 5 circles (top attempts) - show similarity score and coordination
        for i in range(5):
            circle = CoordinationCircle(show_similarity=True, show_coordination=True)
            circles_main_layout.addWidget(circle)
            self.coord_circles.append(circle)

        # Add vertical divider
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        divider.setLineWidth(2)
        circles_main_layout.addWidget(divider)

        # 6th circle (predicted ideal) with label
        predicted_ideal_layout = QVBoxLayout()
        predicted_ideal_label = QLabel("Predicted\nIdeal")
        predicted_ideal_label.setAlignment(Qt.AlignCenter)
        predicted_ideal_label_font = QFont()
        predicted_ideal_label_font.setPointSize(9)
        predicted_ideal_label_font.setBold(True)
        predicted_ideal_label.setFont(predicted_ideal_label_font)
        predicted_ideal_layout.addWidget(predicted_ideal_label)

        self.predicted_ideal_circle = CoordinationCircle(show_similarity=True, show_coordination=False)
        predicted_ideal_layout.addWidget(self.predicted_ideal_circle)

        circles_main_layout.addLayout(predicted_ideal_layout)

        coord_layout.addLayout(circles_main_layout)

        # Legend for circles
        legend_label = QLabel("● Color intensity = Similarity score (black → green)")
        legend_label.setAlignment(Qt.AlignCenter)
        legend_font = QFont()
        legend_font.setPointSize(9)
        legend_label.setFont(legend_font)
        coord_layout.addWidget(legend_label)

        main_layout.addWidget(coord_frame)

        # Status label
        self.status_label = QLabel("Status: Initializing...")
        status_font = QFont()
        status_font.setPointSize(10)
        self.status_label.setFont(status_font)
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

    def _setup_plot(self):
        """Setup stacked plots with 4 channels."""
        # Channel configuration: 0-2 = EEG (Blue, Green, Cyan), 3 = EMG (Red)
        channel_configs = [
            {'color': (0, 100, 255), 'name': 'EEG Ch1', 'width': 1.5},  # Blue
            {'color': (0, 200, 0), 'name': 'EEG Ch2', 'width': 1.5},    # Green
            {'color': (0, 200, 200), 'name': 'EEG Ch3', 'width': 1.5},  # Cyan
            {'color': (255, 0, 0), 'name': 'EMG', 'width': 2.0},        # Red (thicker)
        ]

        # Create stacked subplots
        self.plots = []
        self.curves = []

        for i, config in enumerate(channel_configs):
            # Create subplot
            plot = self.plot_widget.addPlot(row=i, col=0)

            # Configure subplot
            plot.setLabel('left', config['name'], color=config['color'])
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.enableAutoRange(axis='y', enable=True)  # Auto-scale Y-axis
            plot.setMouseEnabled(x=False, y=False)  # Disable mouse interaction

            # Only show X-axis label on bottom plot
            if i == len(channel_configs) - 1:
                plot.setLabel('bottom', 'Time (s)')
            else:
                plot.setLabel('bottom', '')  # Hide x-label for other plots
                # Link X-axes together so they scroll together
                if i > 0:
                    plot.setXLink(self.plots[0])

            # Create curve
            pen = pg.mkPen(color=config['color'], width=config['width'])
            curve = plot.plot(pen=pen)

            self.plots.append(plot)
            self.curves.append(curve)

    def _setup_timer(self):
        """Setup timer for 60 Hz updates."""
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_plot)
        interval_ms = int(1000 / self.update_rate_hz)  # ~17ms for 60Hz
        self.timer.start(interval_ms)

    def add_data(self, samples):
        """
        Add new data samples to the display queue.

        Args:
            samples: Array of shape (n_samples, 4) with voltage data (only first 4 channels used)
        """
        if samples is not None and len(samples) > 0:
            samples = np.asarray(samples)
            if samples.ndim == 2 and samples.shape[1] > 4:
                samples = samples[:, :4]
            self.data_queue.extend(samples)

    def update_coordination_attempts(self, top_attempts, predicted_ideal=None):
        """
        Update the coordination circles display.

        Args:
            top_attempts: List of dicts with 'coordination_index', 'similarity_score',
                         and 'attempt_id' (up to 5 attempts)
            predicted_ideal: Dict with 'coordination_index' and 'similarity_score' for
                           predicted ideal attempt (optional)
        """
        # Sort attempts by coordination index (ascending - lower is better)
        sorted_attempts = sorted(top_attempts, key=lambda x: x.get('coordination_index', float('inf')))

        # Print current attempts
        print("\n=== Coordination Attempts Update ===")
        for i, attempt in enumerate(sorted_attempts, 1):
            coord = attempt.get('coordination_index', 0.0)
            sim = attempt.get('similarity_score', 0.0)
            att_id = attempt.get('attempt_id', '?')
            print(f"  {i}. Attempt #{att_id}: coord={coord:.4f}, similarity={sim:.3f}")

        if predicted_ideal:
            coord = predicted_ideal.get('coordination_index', 0.0)
            sim = predicted_ideal.get('similarity_score', 0.0)
            print(f"  Predicted Ideal: coord={coord:.4f}, similarity={sim:.3f}")
        print("="*36)

        # Update each of the top 5 circles
        for i in range(5):
            if i < len(sorted_attempts):
                attempt = sorted_attempts[i]
                coord_idx = attempt.get('coordination_index', 0.0)
                sim_score = attempt.get('similarity_score', 0.0) or 0.0  # Handle None
                attempt_id = attempt.get('attempt_id', None)
                self.coord_circles[i].set_data(coord_idx, sim_score, attempt_id)
            else:
                # No data for this circle
                self.coord_circles[i].set_data(None, 0.0, None)

        # Update predicted ideal circle (6th circle) - no attempt ID
        if predicted_ideal is not None:
            coord_idx = predicted_ideal.get('coordination_index', 0.0)
            sim_score = predicted_ideal.get('similarity_score', 0.0) or 0.0
            self.predicted_ideal_circle.set_data(coord_idx, sim_score, None)
        else:
            # No predicted ideal data yet
            self.predicted_ideal_circle.set_data(None, 0.0, None)

    def _update_plot(self):
        """Update plot with new data (called at 60 Hz)."""
        if not self.data_queue:
            return

        # Get all queued data
        new_samples = []
        while self.data_queue:
            new_samples.append(self.data_queue.popleft())

        if not new_samples:
            return

        new_samples = np.array(new_samples)
        # Ensure only first 4 channels (10, 4) — ignore extra channels
        if new_samples.ndim == 2 and new_samples.shape[1] > 4:
            new_samples = new_samples[:, :4]
        n_new = len(new_samples)

        # Roll buffers
        self.data_buffer = np.roll(self.data_buffer, -n_new, axis=0)
        self.time_buffer = np.roll(self.time_buffer, -n_new)

        # Add new data
        self.data_buffer[-n_new:] = new_samples

        # Update time axis
        new_times = np.arange(n_new) / self.sampling_rate + self.current_time
        self.time_buffer[-n_new:] = new_times
        self.current_time = new_times[-1] + 1.0 / self.sampling_rate

        # Calculate X-axis range for synchronized scrolling
        if self.current_time > self.window_size_s:
            x_min = self.current_time - self.window_size_s
            x_max = self.current_time
        else:
            x_min = 0
            x_max = self.window_size_s

        # Update each subplot
        for i in range(4):
            x = np.ascontiguousarray(self.time_buffer)
            y = np.ascontiguousarray(self.data_buffer[:, i])
            self.curves[i].setData(x, y)

            # Synchronize X-axis across all plots
            self.plots[i].setXRange(x_min, x_max, padding=0)

            # Auto-scale Y-axis based on visible data
            if self.current_time > 0:
                # Get visible data range
                visible_mask = (x >= x_min) & (x <= x_max)
                visible_y = y[visible_mask]

                if len(visible_y) > 0:
                    y_min = np.min(visible_y)
                    y_max = np.max(visible_y)
                    y_range = y_max - y_min

                    # Add 10% padding
                    if y_range > 0:
                        padding = y_range * 0.1
                        self.plots[i].setYRange(y_min - padding, y_max + padding, padding=0)

        # Update status
        self.status_label.setText(
            f"Status: Running | Time: {self.current_time:.1f}s | "
            f"Buffer: {n_new} samples"
        )

    def closeEvent(self, event):
        """Handle window close."""
        self.timer.stop()
        event.accept()


def demo():
    """Demo with simulated data."""
    app = QApplication(sys.argv)

    gui = LiveGUI(window_size_s=10.0, sampling_rate=200.0, update_rate_hz=60)
    gui.show()

    # Simulate data at 200 Hz
    def add_simulated_data():
        # Generate 10 samples (simulating 50ms of data at 200 Hz)
        t = gui.current_time + np.arange(10) / 200.0
        data = np.column_stack([
            50 * np.sin(2 * np.pi * 10 * t) + np.random.randn(10) * 5,  # EEG 1
            50 * np.sin(2 * np.pi * 12 * t) + np.random.randn(10) * 5,  # EEG 2
            50 * np.sin(2 * np.pi * 8 * t) + np.random.randn(10) * 5,   # EEG 3
            100 * np.sin(2 * np.pi * 20 * t) + np.random.randn(10) * 10, # EMG (higher amplitude)
        ])
        gui.add_data(data)

    # Simulate coordination attempts
    def update_coordination():
        # Simulate 5 attempts in random order (will be sorted by GUI)
        attempts = [
            {'coordination_index': 0.3421, 'similarity_score': 0.45, 'attempt_id': 23},  # Will be 3rd
            {'coordination_index': 0.0845, 'similarity_score': 0.95, 'attempt_id': 45},  # Will be 1st (best)
            {'coordination_index': 0.5890, 'similarity_score': 0.05, 'attempt_id': 12},  # Will be 5th
            {'coordination_index': 0.1523, 'similarity_score': 0.78, 'attempt_id': 38},  # Will be 2nd
            {'coordination_index': 0.4567, 'similarity_score': 0.23, 'attempt_id': 17},  # Will be 4th
        ]
        # Predicted ideal (no coordination displayed, no attempt ID)
        predicted_ideal = {
            'coordination_index': 0.0723,  # Not displayed
            'similarity_score': 1.0  # Perfect match
        }
        gui.update_coordination_attempts(attempts, predicted_ideal)

    # Timer to simulate data arrival at 200 Hz (add batch every 50ms)
    data_timer = QTimer()
    data_timer.timeout.connect(add_simulated_data)
    data_timer.start(50)  # 50ms intervals

    # Update coordination display once after 2 seconds
    QTimer.singleShot(2000, update_coordination)

    sys.exit(app.exec_())


if __name__ == '__main__':
    demo()
