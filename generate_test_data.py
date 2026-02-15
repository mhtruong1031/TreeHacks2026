"""
Generate realistic test data for EEG/EMG pipeline testing.

Creates a 2-minute dataset with:
- 60 movement attempts (0.5 Hz)
- Realistic baseline with low-amplitude noise
- Varying coordination quality across attempts
- Some artifacts for rejection testing
"""

import numpy as np

# Parameters
fs = 200.0  # Sampling rate (Hz)
duration_s = 120.0  # 2 minutes
n_samples = int(fs * duration_s)
n_channels = 4  # EMG, Left EEG, Right EEG, Additional

# Target: 60 attempts over 2 minutes
n_attempts = 60
attempt_interval = duration_s / n_attempts  # ~2 seconds between attempts

print(f"Generating test data:")
print(f"  Duration: {duration_s} seconds ({duration_s/60:.1f} minutes)")
print(f"  Samples: {n_samples:,} @ {fs} Hz")
print(f"  Channels: {n_channels}")
print(f"  Movement attempts: {n_attempts}")
print(f"  Attempt interval: ~{attempt_interval:.1f}s")
print()

# Initialize with baseline noise
np.random.seed(42)
baseline_mean = np.array([0.01, -0.005, 0.002, 0.008])  # Small DC offset
baseline_std = np.array([0.05, 0.045, 0.048, 0.046])    # Realistic noise level

# Generate baseline (low-amplitude Gaussian noise)
data = np.random.randn(n_samples, n_channels) * baseline_std + baseline_mean

# Add 60 movement attempts with varying characteristics
attempt_positions = []
for i in range(n_attempts):
    # Position: evenly spaced with small jitter
    center_pos = int((i + 0.5) * attempt_interval * fs)
    jitter = np.random.randint(-int(0.2 * fs), int(0.2 * fs))  # ±0.2s jitter
    start_pos = max(0, center_pos + jitter)

    # Duration: 200-500ms (40-100 samples)
    duration = np.random.randint(40, 100)
    end_pos = min(n_samples, start_pos + duration)

    if end_pos >= n_samples:
        break

    # Amplitude: varies to create different coordination qualities
    # Good attempts: 0.3-0.5, Poor attempts: 0.15-0.25
    if i % 3 == 0:  # Every 3rd attempt is "good"
        amplitude = np.random.uniform(0.3, 0.5)
        coordination_quality = "good"
    else:
        amplitude = np.random.uniform(0.15, 0.3)
        coordination_quality = "moderate"

    # Create movement pattern (smooth rise and fall)
    movement_length = end_pos - start_pos
    t = np.linspace(0, np.pi, movement_length)
    envelope = np.sin(t)  # Smooth activation envelope

    # Add movement to all channels with slight variations
    for ch in range(n_channels):
        channel_amplitude = amplitude * np.random.uniform(0.8, 1.2)
        movement_signal = channel_amplitude * envelope

        # Add to baseline
        data[start_pos:end_pos, ch] += movement_signal

    attempt_positions.append((start_pos, end_pos, coordination_quality))

print(f"✓ Generated {len(attempt_positions)} movement attempts")

# Add some artifacts (blinks, motion artifacts)
n_artifacts = 10
artifact_positions = np.random.choice(n_samples, n_artifacts, replace=False)
for pos in artifact_positions:
    # Large amplitude spike (should be rejected as artifact)
    artifact_amplitude = np.random.uniform(3.0, 8.0) * np.random.choice([-1, 1])
    artifact_width = np.random.randint(5, 15)  # 25-75ms

    start = max(0, pos - artifact_width // 2)
    end = min(n_samples, pos + artifact_width // 2)

    # Affect random subset of channels
    affected_channels = np.random.choice(n_channels, size=np.random.randint(1, n_channels+1), replace=False)
    for ch in affected_channels:
        data[start:end, ch] += artifact_amplitude * np.exp(-np.linspace(-2, 2, end-start)**2)

print(f"✓ Added {n_artifacts} artifacts")

# Add low-frequency drift (realistic baseline variation)
drift_freq = 0.02  # Hz (50 second period)
t = np.arange(n_samples) / fs
drift = 0.02 * np.sin(2 * np.pi * drift_freq * t)
data += drift[:, np.newaxis]

print(f"✓ Added baseline drift")

# Save to CSV
output_file = 'test_data.csv'
np.savetxt(output_file, data, delimiter=',', fmt='%.6f')
print()
print(f"✓ Saved to {output_file}")
print()

# Summary statistics
print("Data Statistics:")
print(f"  Channel means: {np.mean(data, axis=0)}")
print(f"  Channel stds:  {np.std(data, axis=0)}")
print(f"  Min value: {np.min(data):.4f}")
print(f"  Max value: {np.max(data):.4f}")
print()

# Attempt statistics
good_attempts = sum(1 for _, _, q in attempt_positions if q == "good")
moderate_attempts = len(attempt_positions) - good_attempts
print("Movement Attempt Distribution:")
print(f"  Good coordination: {good_attempts} attempts")
print(f"  Moderate coordination: {moderate_attempts} attempts")
print(f"  Total: {len(attempt_positions)} attempts")
print()

# Expected detection
expected_detectable = sum(1 for start, end, q in attempt_positions
                         if np.max(np.abs(data[start:end])) > 0.2)
print(f"Expected detectable (with threshold=0.2): {expected_detectable} attempts")
print(f"Expected detectable (with adaptive ~0.25): ~{len(attempt_positions)} attempts")
print()

# Timeline preview
print("Timeline Preview (first 20 seconds):")
for start, end, quality in attempt_positions[:10]:
    time_s = start / fs
    duration_ms = (end - start) / fs * 1000
    peak_amplitude = np.max(np.abs(data[start:end]))
    print(f"  {time_s:5.1f}s: {duration_ms:3.0f}ms burst, "
          f"peak={peak_amplitude:.3f}, quality={quality}")
print()

print("✅ Test data generation complete!")
print(f"   Run with: python main_runtime.py --simulate {output_file} --speed 0 --calibration-time 5")
