# Quick Start Guide - Baseline Calibration System

## Installation

No additional dependencies required! The system uses existing packages:
- numpy
- scipy
- torch
- (matplotlib for plotting - optional)

## Usage

### 1. Live EEG/EMG Session (OpenBCI)

**Step 1: Start OpenBCI GUI and configure UDP streaming**
- Set IP: 127.0.0.1 (or 0.0.0.0 for network streaming)
- Set Port: 12345
- Start streaming

**Step 2: Run the runtime system**
```bash
python main_runtime.py --live --calibration-time 60
```

**What happens:**
1. **Calibration Phase (60 seconds):**
   - Sit still and relax
   - System collects baseline data
   - Computes channel statistics

2. **Streaming Phase (continuous):**
   - Processes packets in real-time
   - Detects movement attempts with adaptive thresholds
   - Rejects artifacts automatically
   - Displays progress every 1000 packets

**Output example:**
```
PHASE 1: BASELINE CALIBRATION
Please sit still and relax for 60 seconds...
[100.0%] 12,000 / 12,000 samples | 200.0 Hz | 60.0s
✓ Calibration complete

PHASE 2: STREAMING ANALYSIS
[    5.0s] Packets:   1,000 | Rate: 200.0 Hz | Attempts: 3 | Model: Pending (3/30)
  Top attempts:
    1. Coord: 0.4609 | Similarity: 0.0000
    2. Coord: 0.3930 | Similarity: 0.2121
```

### 2. Testing with Simulated Data

**For fast testing (unlimited speed):**
```bash
python main_runtime.py --simulate test_data.csv --speed 0 --calibration-time 5
```

**For realistic simulation (real-time):**
```bash
python main_runtime.py --simulate data.csv --speed 1.0
```

### 3. Advanced Options

**Enable spatial whitening:**
```bash
python main_runtime.py --live --whitening
```

**Adjust adaptive threshold sensitivity:**
```bash
python main_runtime.py --live --threshold-n-std 3.0  # More conservative (3σ)
```

**Skip calibration (legacy fixed threshold mode):**
```bash
python main_runtime.py --simulate data.csv --skip-calibration
```

**Custom UDP configuration:**
```bash
python main_runtime.py --live --ip 192.168.1.100 --port 8080
```

## Command-Line Options

```
Required (choose one):
  --live                    Use live UDP data from OpenBCI
  --simulate CSV_PATH       Use CSV file for testing

UDP options:
  --ip IP                   UDP IP address (default: 0.0.0.0)
  --port PORT               UDP port (default: 12345)

Simulation options:
  --speed SPEED             Playback speed (0=unlimited, 1=real-time, 10=10x)

Calibration options:
  --calibration-time SECS   Calibration duration (default: 60s)
  --skip-calibration        Use fixed thresholds instead
  --whitening               Enable spatial whitening
  --threshold-n-std N       Adaptive threshold multiplier (default: 2.5)

System options:
  --fs HZ                   Sampling rate (default: 200 Hz)
```

## Programmatic Usage

### With Calibration (Recommended)

```python
import numpy as np
from analysis.BaselineCalibration import BaselineCalibration
from analysis.MainPipeline import MainPipeline

# Step 1: Collect baseline data (60 seconds, 4 channels @ 200 Hz)
rest_data = []  # Collect from your data source
# ... (collect 12,000 samples)
rest_data_array = np.array(rest_data)  # Shape: (12000, 4)

# Step 2: Calibrate
calib = BaselineCalibration(fs=200.0)
calib.calibrate(rest_data_array)

# Step 3: Create pipeline with calibration
pipeline = MainPipeline(
    window_size_s=0.2,
    calibration=calib,
    use_whitening=False,
    adaptive_threshold_n_std=2.5
)

# Step 4: Process packets
for packet in data_stream:  # packet shape: (4,)
    pipeline.run(packet)

# Step 5: Monitor status
info = pipeline.get_calibration_info()
print(f"Artifacts rejected: {info['artifacts_rejected']}")
print(f"Artifact rate: {info['artifact_rate']*100:.1f}%")
```

### Without Calibration (Legacy Mode)

```python
from analysis.MainPipeline import MainPipeline

# Old API still works
pipeline = MainPipeline(window_size_s=0.2, activation_threshold=0.2)

for packet in data_stream:
    pipeline.run(packet)
```

### Using DataSource Abstraction

```python
from data_sources import UDPDataSource, SimulatedDataSource

# Option 1: Live UDP
source = UDPDataSource(ip="0.0.0.0", port=12345, timeout=1.0)

# Option 2: Simulated CSV
source = SimulatedDataSource("data.csv", fs=200.0, speed=1.0)

# Both use the same interface
while source.is_running():
    packet = source.get_packet()  # Returns np.ndarray or None
    if packet is not None:
        pipeline.run(packet)

source.close()
```

## Understanding the Output

### Calibration Statistics
```
Channel means: [-0.0038, 0.0020, -0.0028, 0.0066]  # DC offset per channel
Channel stds:  [0.0924, 0.0829, 0.0828, 0.0837]     # Noise level per channel
Thresholds:    [0.2273, 0.2093, 0.2041, 0.2159]     # Activation thresholds
```

### Streaming Statistics
```
[    5.0s]      # Elapsed time
Packets: 1,000  # Total packets processed
Rate: 200.0 Hz  # Current throughput
Attempts: 3     # Movement attempts detected
Model: Pending  # Prediction model status
T  S  P         # Async operations: Training, Similarity, Prediction (⏳ = active)
Artifacts: 56   # Packets rejected as artifacts
```

### Top Attempts
```
1. Coord: 0.4609 | Similarity: 0.0000  # Best attempt (lowest coordination index)
2. Coord: 0.3930 | Similarity: 0.2121  # Second best
3. Coord: 0.3007 | Similarity: 0.2391  # Third best
```
- **Coordination Index:** Lower = better coordinated movement
- **Similarity Score:** Higher = more similar to reference attempt

## Troubleshooting

### "No data received after 100 attempts"
- **Cause:** UDP data not streaming or wrong IP/port
- **Fix:** Check OpenBCI GUI networking settings
- **Fix:** Try `--ip 0.0.0.0` to listen on all interfaces

### High artifact rejection rate (>10%)
- **Cause:** Noisy environment or poor electrode contact
- **Fix:** Improve electrode placement
- **Fix:** Increase threshold: `--threshold-n-std 3.0` or `4.0`

### Low artifact rejection rate (0%)
- **Cause:** Very clean baseline or threshold too high
- **Status:** Normal if baseline is clean
- **Check:** Verify artifacts are actually present in data

### Model never trains
- **Cause:** Not enough movement attempts detected
- **Status:** Need 30 attempts before initial training
- **Fix:** Lower activation threshold or perform more movements

### Calibration fails
- **Cause:** Not enough calibration data collected
- **Fix:** Ensure data source is streaming during calibration
- **Fix:** Increase `--calibration-time` if data rate is slow

## Performance Tips

1. **Fast Testing:** Use `--speed 0` with simulated data for unlimited speed
2. **Production:** Use default `--speed 1.0` for accurate timing
3. **Memory:** System automatically trims old data (default: 20 min @ 100Hz)
4. **Artifacts:** High rejection rate (>20%) indicates electrode issues

## Next Steps

1. **Test with your data:** `python main_runtime.py --simulate your_data.csv --speed 0`
2. **Try live streaming:** `python main_runtime.py --live`
3. **Experiment with thresholds:** Try different `--threshold-n-std` values
4. **Enable whitening:** Add `--whitening` for better artifact handling

## Support

For issues or questions:
- Check `IMPLEMENTATION_SUMMARY.md` for technical details
- Run test suite: `python test_comprehensive.py`
- Verify backward compatibility: `python test_backward_compat.py`
