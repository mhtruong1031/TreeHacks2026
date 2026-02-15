"""
Test backward compatibility: old MainPipeline code should still work.
"""

import numpy as np
from analysis.MainPipeline import MainPipeline

# Test 1: Old initialization (without calibration)
print("Test 1: Old-style MainPipeline initialization")
pipeline = MainPipeline(window_size_s=0.2, activation_threshold=0.2)
print("✓ Pipeline created successfully")

# Test 2: Process some packets
print("\nTest 2: Process packets without calibration")
for i in range(100):
    packet = np.random.randn(4) * 0.1
    pipeline.run(packet)
print(f"✓ Processed 100 packets")

# Test 3: Check that it's using fixed threshold mode
info = pipeline.get_calibration_info()
print(f"\nTest 3: Verify fixed threshold mode")
print(f"  Calibrated: {info['calibrated']}")
print(f"  Mode: {info['mode']}")
print(f"  Fixed threshold: {info['fixed_threshold']}")
assert info['calibrated'] == False
assert info['mode'] == 'fixed_threshold'
print("✓ Fixed threshold mode confirmed")

# Test 4: With calibration
print("\nTest 4: New-style initialization with calibration")
from analysis.BaselineCalibration import BaselineCalibration

# Create calibration
calib = BaselineCalibration(fs=200.0)
rest_data = np.random.randn(12000, 4) * 0.05  # 60 seconds of rest data
calib.calibrate(rest_data)

# Create pipeline with calibration
pipeline2 = MainPipeline(calibration=calib, adaptive_threshold_n_std=2.5)
print("✓ Pipeline with calibration created")

# Process packets
for i in range(100):
    packet = np.random.randn(4) * 0.1
    pipeline2.run(packet)
print(f"✓ Processed 100 packets with calibration")

# Check adaptive mode
info2 = pipeline2.get_calibration_info()
print(f"\nTest 5: Verify adaptive threshold mode")
print(f"  Calibrated: {info2['calibrated']}")
print(f"  Mode: {info2['mode']}")
print(f"  Adaptive thresholds: {info2['adaptive_thresholds']}")
assert info2['calibrated'] == True
assert info2['mode'] == 'adaptive'
print("✓ Adaptive threshold mode confirmed")

# Test 6: Artifact rejection
print("\nTest 6: Test artifact rejection")
artifact_packet = np.array([5.0, 5.0, 5.0, 5.0])  # Large artifact
normal_packet = np.array([0.05, 0.05, 0.05, 0.05])

pipeline2.run(artifact_packet)
artifacts_1 = pipeline2.artifact_count

pipeline2.run(normal_packet)
artifacts_2 = pipeline2.artifact_count

print(f"  Artifacts after large packet: {artifacts_1}")
print(f"  Artifacts after normal packet: {artifacts_2}")
assert artifacts_1 > 0, "Large packet should be rejected as artifact"
print("✓ Artifact rejection working")

print("\n" + "=" * 60)
print("✅ All backward compatibility tests passed!")
print("=" * 60)
