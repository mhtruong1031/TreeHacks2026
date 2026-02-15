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
UDP packets (OpenBCI) → udp_receiver.py → MainPipeline
                                              ↓
                                        Sliding window buffer (deque)
                                              ↓
                                        PreprocessingPipeline
                                              ↓
                                        Activation detection (threshold-based)
                                              ↓
                                        PresentPipeline (coordination index)
                                              ↓
                                        MaxNCoordCache (priority queue of attempts)
                                              ↓
                                        PredictionPipeline (after threshold samples)
```

### Key Components

- **MainPipeline**: Orchestrator with sliding window buffer; detects activations when signal exceeds threshold
- **MaxNCoordCache**: Max-heap that stores movement attempts ranked by coordination index (bucketed by epsilon for recency tie-breaking)
- **BaselineCalibration**: Computes noise parameters from resting-state recording for artifact removal (blink templates, noise PSD, covariance matrix)
- **StreamSimulator**: Replays CSV data as simulated real-time stream for testing

## Development Commands

### Running Live EEG Receiver

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
# Simulate EEG stream from CSV
python -m simulation.simulate_eeg_stream --csv simulation/dataset/rest/trial_001.csv --speed 10

# Test full preprocessing pipeline
python -m simulation.test_stream_pipeline --csv simulation/dataset/rest/trial_001.csv --window 1.0 --speed 0

# Generate synthetic dataset
python -m simulation.generate_synthetic_dataset --subjects 10 --trials 30
```

### Running the Main Pipeline

The main pipeline is designed to be instantiated and fed packets:

```python
from analysis.MainPipeline import MainPipeline

pipeline = MainPipeline(window_size_s=0.2, activation_threshold=0.2)
# Feed packets one at a time
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

### Channel Convention (post background-subtraction)
- Channel 0: EMG (muscle activity)
- Channel 1: Left head EEG
- Channel 2: Right head EEG

### Key Parameters
- **Window size**: Default 0.2s (40 samples @ 200 Hz) in MainPipeline
- **Activation threshold**: Default 0.2 (voltage threshold for movement detection)
- **Activation buffer**: 3 consecutive points above threshold required
- **Prediction data threshold**: 30 high-quality attempts before training initial model
- **Cache epsilon**: 0.1 (coordination index bucketing for recency tie-breaking)

## Dataset Structure

`simulation/dataset/` contains synthetic data organized by movement class:
- **rest**: Baseline with alpha rhythm
- **motor_imagery**: Imagined movement (mu/alpha suppression)
- **wrist_flex_ext**: Alternating wrist movements
- **grip_release**: Grip and release patterns
- **cocontraction**: Simultaneous antagonist muscle activation

Each trial stored as NPZ with metadata in `manifest.csv` and class mapping in `class_map.json`.

## Dependencies

Key Python packages (inferred from imports):
- `numpy` - Array operations
- `scipy` - Signal processing (butter, filtfilt, decimate, resample, welch)
- `matplotlib` - Plotting (optional, for --plot flag)
- `torch` - Neural network model
- `tslearn` - Dynamic time warping for similarity scoring

## Testing Strategy

1. **Unit-level**: Test individual filters using `simulation/compare_filter_configs.py`
2. **Integration**: Use `simulation/test_stream_pipeline.py` to test preprocessing end-to-end
3. **Simulation**: Use `StreamSimulator` to replay real or synthetic data at controllable speed
4. **Live**: Run `udp_receiver.py` with OpenBCI GUI for actual hardware testing

## Architecture Notes

- **MainPipeline uses deque for sliding window**: Automatically drops oldest samples when full (O(1) operations)
- **Two data buffers**:
  - `self.data`: Raw window buffer (deque, only keeps `window_size_samples`)
  - `self.processed_data`: Full history list (needed for activation window slicing)
- **Activation windows**: Stored as `(start_idx, end_idx)` tuples referencing indices in `processed_data`
- **Cache behavior**: MaxNCoordCache maintains heap of best attempts; nodes can be updated with similarity scores post-insertion
- **Model training**: Initial model trained on first N attempts (batch), then incremental updates (online) for new attempts
- **Filters use SOS format**: Second-order sections for numerical stability (`scipy.signal.butter(..., output="sos")`)
