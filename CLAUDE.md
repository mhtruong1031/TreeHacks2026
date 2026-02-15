# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time EEG/EMG signal processing system for motor coordination analysis and prediction. The system receives live brain/muscle activity data (via OpenBCI GUI over UDP), processes it through multiple pipeline stages, and provides coordination metrics and predictions using a PyTorch-based neural network.

## Core Architecture

### Three-Stage Pipeline Architecture

The system is organized as a sequential processing pipeline orchestrated by `MainPipeline`:

1. **Preprocessing** (`PreprocessingPipeline`)
   - Downsamples from 200 Hz to 100 Hz (configurable via `target_fs`)
   - Applies bandpass filter (0.5-45 Hz) to remove DC drift and high-frequency noise
   - Applies lowpass filter (30 Hz) for blink artifact attenuation
   - Applies bandstop filter (0.5 Hz ± 0.2 Hz) for sweat artifact removal

2. **Present Analysis** (`PresentPipeline`)
   - Computes **coordination index** via SVD energy analysis and bimodality coefficient
   - Coordination index measures movement quality: distance from ideal (1.0, 0.0) in (SVD_energy, bimodality) space
   - Calculates **similarity scores** using dynamic time warping (DTW) between movement attempts
   - Lower coordination index = better coordinated movement

3. **Prediction** (`PredictionPipeline`)
   - PyTorch GRU-based sequential model (utils/Model.py)
   - Predicts ideal movement patterns based on historical high-quality attempts
   - Trained incrementally as new data arrives (online learning)

### Data Flow

```
UDP packets (OpenBCI) → MainPipeline (200 Hz)
                            ↓
                      Sliding window buffer (deque)
                            ↓
                      PreprocessingPipeline
                            ↓
                      Activation detection (dual-buffer: 5-sample hysteresis)
                            ↓
                      PresentPipeline (coordination index) [FAST: <1ms]
                            ↓
                      MaxNCoordCache (add node) [FAST: <1ms]
                            ↓
                      ┌─────────────────────────────┬─────────────────────────────┐
                      ↓                             ↓                             ↓
           [Background Thread 1]         [Background Thread 2]         [Main Thread continues]
              Training/Prediction            Similarity Scores              Next packet ready
              (100ms-1s)                     (50-200ms)                     No blocking!
                      ↓                             ↓
              on_prediction_ready()         on_similarity_ready()
                      └─────────────────────────────┘
                                    ↓
                            GUI updates immediately
```

### Key Components

- **MainPipeline**: Orchestrator with sliding window buffer; detects activations using dual-buffer (5-sample hysteresis)
- **MaxNCoordCache**: Max-heap that stores movement attempts ranked by coordination index (bucketed by epsilon for recency tie-breaking)
- **BaselineCalibration**: Computes noise parameters from resting-state recording for artifact removal (blink templates, noise PSD, covariance matrix)
- **LiveGUI**: PyQtGraph-based real-time visualization with 4 stacked subplots, auto-scaling, and coordination circles display
- **GUIController**: Connects MainPipeline to LiveGUI with background processing and immediate callback updates

### Data Source Abstraction

The `data_sources/` package provides unified interfaces for different data inputs:

- **DataSource** (base class): Abstract interface with `get_packet()` and `is_running()`
- **UDPDataSource**: Receives live packets from OpenBCI GUI via UDP
- **SimulatedDataSource**: Replays CSV files with configurable speed (producer-consumer pattern)

All data sources provide packets as `(n_channels,)` arrays through the same interface.

### GUI Architecture (run_live_gui.py)

The GUI provides real-time visualization of EEG/EMG signals and coordination metrics:

**Threading Model (Non-Blocking):**
- **Qt Main Thread**: GUI updates at 60 Hz, user interactions, displays coordination circles
- **Background Worker Thread**: Runs MainPipeline at 200 Hz (fast path: <1ms per packet)
- **Async Training Thread**: Model training/prediction (100ms-1s) - daemon, non-blocking
- **Async Similarity Thread**: DTW similarity calculations (50-200ms) - daemon, non-blocking
- **Communication**: Thread-safe callbacks (`on_prediction_ready`, `on_similarity_ready`) trigger immediate GUI updates

**Key GUI Components:**
- **LiveGUI** (`gui/live_gui.py`): Main window with 4 stacked subplots + coordination display
  - 4 vertically stacked subplots (one per channel) with synchronized X-axis
  - Independent auto-scaling Y-axis per channel (based on visible data range)
  - 10-second rolling window at 200 Hz sampling rate
  - 60 Hz display refresh (QTimer at 17ms intervals)
- **CoordinationCircle**: Widget displaying attempt quality with color-coded similarity
  - Top 5 circles: Best coordinated attempts (sorted ascending by coordination index)
  - 6th circle: Predicted ideal pattern from model
  - Color: Black (0.0) → Green (1.0) based on similarity score
  - Labels: "Similarity:" above circle, "Coordination:" below with index value
  - Attempt numbers displayed at top of each circle
- **GUIController** (`run_live_gui.py`): Connects pipeline to GUI
  - Two-phase execution: calibration (60s default) → streaming
  - Registers callbacks for immediate updates when predictions complete
  - Updates coordination display every 0.5s + on-demand via callbacks

**Async Processing (Critical for Performance):**
- Main data flow NEVER blocks - training/prediction happen in background
- `MainPipeline._async_train_and_predict()`: Trains model + generates prediction in separate thread
- `MainPipeline._async_update_similarity()`: Calculates DTW similarities in separate thread
- Both use thread locks (`cache_lock`, `model_lock`) for safety
- Callbacks fire immediately when operations complete (no 0.5s wait)

**GUI Performance Notes:**
- OpenGL disabled for macOS stability (`pg.setConfigOption('useOpenGL', False)`)
- Anti-aliasing disabled to prevent crashes
- Data arrays converted to contiguous for PyQt compatibility (`np.ascontiguousarray`)
- Never use `--speed 0` with GUI - causes race conditions (use speed ≥ 1.0)

## Development Commands

### Live GUI System (Recommended)

The `run_live_gui.py` provides real-time visualization with coordination circles:

```bash
# Live GUI with simulated data
python run_live_gui.py --simulate test_data.csv --speed 1.0

# Live GUI with UDP data source
python run_live_gui.py --live --ip 0.0.0.0 --port 12345 --calibration-time 60

# Skip calibration for quick testing
python run_live_gui.py --simulate test_data.csv --speed 1.0 --skip-calibration

# Fast testing (10x speed)
python run_live_gui.py --simulate test_data.csv --speed 10 --calibration-time 5
```

**Quick GUI Test (10 seconds with mock data):**
```bash
python test_live_gui.py  # Auto-closes after 10s
```

**Important GUI Notes:**
- **Never use `--speed 0` with GUI** - causes threading issues on macOS
- Use `--speed 1.0` or higher for GUI mode (real-time or faster)
- GUI requires: `pip install PyQt5 pyqtgraph`
- Calibration can be skipped with `--skip-calibration` flag
- Display shows: 4 stacked plots + top 5 coordination circles + predicted ideal circle

### Running Live EEG Receiver (Legacy)

```bash
# Basic receiver (console output only)
python udp_receiver.py

# With plotting and custom parameters
python udp_receiver.py --plot --fs 250 --ip 0.0.0.0 --port 12345

# Plot FFT up to specific frequency
python udp_receiver.py --plot --fs 250 --max-freq 60
```

### Testing with Simulated Data

```bash
# Generate test data
python generate_test_data.py

# Simulate EEG stream from CSV
python -m simulation.simulate_eeg_stream --csv simulation/dataset/rest/trial_001.csv --speed 10

# Test full preprocessing pipeline
python -m simulation.test_stream_pipeline --csv simulation/dataset/rest/trial_001.csv --window 1.0 --speed 0

# Generate synthetic dataset
python -m simulation.generate_synthetic_dataset --subjects 10 --trials 30
```

### GUI Testing & Debugging

```bash
# Quick 10-second test with simulated data (recommended)
python test_live_gui.py

# Test with custom calibration time
python run_live_gui.py --simulate test_data.csv --speed 1.0 --calibration-time 5 --skip-calibration

# Check if coordination circles update (watch console for [DEBUG] output)
python run_live_gui.py --simulate test_data.csv --speed 1.0 --skip-calibration
```

### Running the Main Pipeline Programmatically

```python
from analysis.MainPipeline import MainPipeline
from data_sources import SimulatedDataSource

# Create data source
data_source = SimulatedDataSource('test_data.csv', fs=200.0, speed=1.0)

# Create pipeline
pipeline = MainPipeline(window_size_s=0.2, activation_threshold=0.2)

# Process packets
while data_source.is_running():
    packet = data_source.get_packet()
    if packet is not None:
        pipeline.run(packet)  # packet shape: (n_channels,)
```

## Important Constants & Conventions

### Sampling Rates
- **Raw data**: 200 Hz (OpenBCI default)
- **After preprocessing**: 100 Hz (default `target_fs`, configurable)
- UDP receiver supports other rates via `--fs` flag

### Data Shape Conventions
- **Raw packets**: `(n_channels,)` - single time point
- **Windows for preprocessing**: `(n_samples, 1 + n_channels)` where column 0 is time, columns 1+ are voltage
- **After preprocessing**: Same column convention, fewer rows (downsampled)
- **Processed data buffer**: `(n_samples, n_features)` - time × features

### Channel Convention (GUI Display Order)
- **Channel 0** (EEG Ch1): Brain activity - Blue (RGB: 0, 100, 255) - Top subplot
- **Channel 1** (EEG Ch2): Brain activity - Green (RGB: 0, 200, 0) - Second subplot
- **Channel 2** (EEG Ch3): Brain activity - Cyan (RGB: 0, 200, 200) - Third subplot
- **Channel 3** (EMG): Muscle activity - Red (RGB: 255, 0, 0), thicker line - Bottom subplot

**Notes:**
- Model expects 4-channel input. `processed_data` items are `(4,)` arrays (voltage only, no time column)
- GUI always uses first 4 channels - extra channels silently ignored
- Each channel has independent Y-axis auto-scaling based on visible data range
- Subplots synchronized on X-axis (time) for aligned scrolling

### Key Parameters
- **Window size**: Default 0.2s (40 samples @ 200 Hz) in MainPipeline
- **Activation threshold**: Default 0.2 (voltage threshold for movement detection)
- **Activation buffer**: 5 consecutive points (dual-buffer at both start and end)
  - **Start buffer**: 5 consecutive high points to START activation window
  - **End buffer**: 5 consecutive low points to END activation window
  - **Effect**: 5-sample hysteresis prevents fragmentation from brief signal fluctuations
- **Prediction data threshold**: 30 high-quality attempts before training initial model
- **Cache epsilon**: 0.1 (coordination index bucketing for recency tie-breaking)
- **GUI update rate**: 60 Hz for plot refresh, immediate updates via callbacks for predictions
- **Coordination display**: Updates every 0.5s + immediately when prediction/similarity ready

## Dataset Structure

`simulation/dataset/` contains synthetic data organized by movement class:
- **rest**: Baseline with alpha rhythm
- **motor_imagery**: Imagined movement (mu/alpha suppression)
- **wrist_flex_ext**: Alternating wrist movements
- **grip_release**: Grip and release patterns
- **cocontraction**: Simultaneous antagonist muscle activation

Each trial stored as NPZ with metadata in `manifest.csv` and class mapping in `class_map.json`.

## Dependencies

**Core (Required):**
- `numpy` - Array operations
- `scipy` - Signal processing (butter, filtfilt, decimate, resample, welch)
- `torch` - PyTorch GRU neural network model
- `tslearn` - Dynamic time warping for similarity scoring

**Optional:**
- `matplotlib` - Legacy plotting (for `udp_receiver.py --plot`)
- `PyQt5` - GUI framework (required for `--gui` mode)
- `pyqtgraph` - Real-time plotting library (required for `--gui` mode)

**Installation:**
```bash
# Core dependencies (console mode)
pip install numpy scipy torch tslearn

# GUI dependencies (optional)
pip install PyQt5 pyqtgraph
```

## Testing Strategy

1. **Unit-level**: Test individual filters using `simulation/compare_filter_configs.py`
2. **Integration**: Use `simulation/test_stream_pipeline.py` to test preprocessing end-to-end
3. **Simulation**: Use `StreamSimulator` to replay real or synthetic data at controllable speed
4. **Live**: Run `udp_receiver.py` with OpenBCI GUI for actual hardware testing

## Architecture Notes

### Data Structures
- **MainPipeline uses deque for sliding window**: Automatically drops oldest samples when full (O(1) operations)
- **Two data buffers**:
  - `self.data`: Raw window buffer (deque, only keeps `window_size_samples`)
  - `self.processed_data`: Full history list (needed for activation window slicing)
- **Activation windows**: Stored as `(start_idx, end_idx)` tuples referencing indices in `processed_data`
- **Cache behavior**: MaxNCoordCache maintains heap of best attempts; nodes can be updated with similarity scores post-insertion

### Dual-Buffer Activation Detection
- **State tracking**: `in_activation_window`, `activation_count`, `deactivation_count`, `activation_start_idx`
- **Start logic**: Signal must stay above threshold for 5 consecutive samples to start window
- **End logic**: Signal must stay below threshold for 5 consecutive samples to end window
- **Hysteresis effect**: Prevents fragmentation - brief signal drops don't split movements into multiple windows
- **Window boundaries**: Start at first high point (after buffer fills), end at first low point (after buffer fills)

### Async Processing (Non-Blocking Data Flow)
- **Main thread**: Preprocessing + coordination index only (~1ms per packet at 200 Hz)
- **Training thread**: `_async_train_and_predict()` runs in background daemon thread
  - Initial training: Batch on first 30 attempts
  - Incremental updates: Online learning on each new attempt
  - Prediction: Generate ideal pattern, calculate similarity to top 5 attempts
  - Callback: `on_prediction_ready()` triggers immediate GUI update
- **Similarity thread**: `_async_update_similarity()` runs DTW calculations in background
  - Compares current attempt to top 5 cached attempts
  - Uses interpolation + Euclidean distance for speed
  - Callback: `on_similarity_ready()` triggers immediate GUI update
- **Thread safety**: `cache_lock` and `model_lock` prevent race conditions
- **Flags**: `training_in_progress`, `similarity_in_progress` prevent duplicate operations

### Model and Scoring
- **Model training**: Initial model trained on first N attempts (batch), then incremental updates (online)
- **Predicted ideal similarity**: Calculated by comparing prediction to top 5 actual attempts (mean similarity)
- **Similarity metric**: Fast interpolation-based Euclidean distance (O(n)) via `PresentPipeline.get_similarity_score()`
  - Interpolates sequences to common length (handles variable-length attempts)
  - Computes Euclidean distance between interpolated signals
  - Returns value where 0.0 = completely dissimilar, 1.0 = identical
- **Coordination index**: Distance from ideal (1.0, 0.0) in (SVD_energy, bimodality) space
  - Calculated in `PresentPipeline.get_coordination_index()`
  - Lower values = better coordination quality
  - Ranges typically 0.05-0.8 (0.0-0.2 = excellent, 0.2-0.4 = good, 0.4+ = fair/poor)
- **Filters use SOS format**: Second-order sections for numerical stability (`scipy.signal.butter(..., output="sos")`)

### Critical Implementation Details

**Thread Safety:**
- Always acquire `cache_lock` when accessing `max_n_coord_cache` from any thread
- Always acquire `model_lock` when training or predicting with model
- GUI callbacks may be called from background threads - ensure thread-safe operations

**Callback Registration (in GUIController):**
```python
self.pipeline.on_prediction_ready = self._on_prediction_ready
self.pipeline.on_similarity_ready = self._on_similarity_ready
```

**Activation Detection State:**
- Do NOT modify activation detection logic without understanding state machine
- State variables: `in_activation_window`, `activation_count`, `deactivation_count`, `activation_start_idx`
- Both start and end require 5-sample buffers - removing either causes instability

**GUI Data Compatibility:**
- Always use `np.ascontiguousarray()` before passing to PyQt/pyqtgraph
- GUI expects shape `(n_samples, 4)` - slice to first 4 channels if more exist
- Never pass None or empty arrays to `LiveGUI.add_data()`

## Coordination Display (GUI Feature)

The GUI shows 6 circles below the main plot representing movement quality:

**Top 5 Circles (Actual Attempts):**
- Sorted by coordination index (ascending - lower = better quality)
- Circle color: Black (0.0) → Green (1.0) based on similarity score
  - Green intensity: `int(255 * similarity_score * 0.65)` for visibility
- Text display:
  - Top: "Attempt #X" (attempt number from ranking)
  - Inside: Similarity score (white text)
  - Below: "Coordination:" label + coordination index (4 decimals)

**6th Circle (Predicted Ideal):**
- Separated by vertical divider
- Shows model's prediction of ideal movement pattern
- Similarity calculated by comparing prediction to top 5 actual attempts (mean)
- Color intensity reflects how well prediction matches actual attempts
- No coordination index displayed (always 0 for ideal)
- No attempt number displayed

**Interpretation:**
- **Bright green circles**: High-quality attempts similar to ideal pattern
- **Dark/black circles**: Poor coordination or dissimilar to ideal
- **Coordination index**: Lower is better (distance from ideal in feature space)
- **Similarity score**: Higher is better (pattern similarity via interpolated Euclidean distance)

**Update Behavior:**
- Periodic: Every 0.5 seconds (100 packets @ 200 Hz)
- Immediate: When prediction completes or similarity scores update (via callbacks)
- Printed to console with `[DEBUG]` tags showing current values

## Common Issues & Troubleshooting

### GUI Mode Issues (macOS)

**Bus Error / Segmentation Fault:**
- **Cause**: Speed=0 with GUI causes threading race conditions
- **Solution**: Use `--speed 1.0` or higher
- **Example**: `python run_live_gui.py --simulate test_data.csv --speed 1.0`

**Coordination Circles Not Updating:**
- Check console for `[DEBUG]` messages showing retrieved nodes
- Verify attempts are being collected: Look for "Activation window START/END" messages
- Ensure prediction threshold reached (30 attempts minimum for model training)
- Check similarity scores are calculated (not None or 0.0 for all attempts)

**Predicted Ideal Shows Black/No Color:**
- Means similarity to actual attempts is low (prediction doesn't match movements)
- Normal in early training - improves as model learns from more attempts
- Check `[DEBUG] Predicted ideal: coord=X, sim=Y` to see actual values

**Plot Widget Crashes:**
- Ensure OpenGL is disabled (already configured in `LiveGUI`)
- Anti-aliasing disabled by default for stability
- Test with: `python test_live_gui.py` (10-second quick test)

**Missing Dependencies:**
```bash
pip install PyQt5 pyqtgraph numpy scipy torch tslearn
```

### Data Processing Issues

**Calibration Errors:**
- Ensure calibration time provides enough samples (1s = 200 samples @ 200 Hz)
- Use `--skip-calibration` for testing without calibration phase
- Check that data source has sufficient data for calibration period

**Worker Thread Hangs:**
- Check data source is providing packets (`data_source.is_running()`)
- For SimulatedDataSource, ensure CSV file exists and is readable
- Add debug prints to track packet flow

## File Organization

```
TreeHacks2026/
├── run_live_gui.py           # Live GUI entry point (recommended)
├── test_live_gui.py          # Quick 10-second GUI test
├── analysis/                 # Core pipeline components
│   ├── MainPipeline.py       # Main orchestrator with async processing
│   ├── PreprocessingPipeline.py
│   ├── PresentPipeline.py    # Coordination index + similarity scores
│   ├── PredictionPipeline.py # PyTorch GRU model
│   └── BaselineCalibration.py
├── data_sources/             # Data input abstraction
│   ├── base.py              # DataSource interface
│   ├── udp_source.py        # Live UDP input (OpenBCI)
│   └── simulated_source.py  # CSV file replay with controllable speed
├── gui/                      # GUI components
│   ├── live_gui.py          # Main window with stacked plots + coordination circles
│   └── __init__.py          # Package exports
├── utils/                    # Utilities
│   ├── Model.py             # PyTorch GRU sequential model
│   └── MaxCoordCache.py     # Max-heap priority queue for attempts
└── simulation/              # Testing utilities and dataset generation
```

**Key Files:**
- `run_live_gui.py`: Main entry point - integrates MainPipeline with LiveGUI
- `gui/live_gui.py`: Contains `LiveGUI` (main window) and `CoordinationCircle` (attempt display widget)
- `analysis/MainPipeline.py`: Core orchestrator with async training/prediction methods
- `test_live_gui.py`: Quick test with simulated data (auto-closes after 10s)
