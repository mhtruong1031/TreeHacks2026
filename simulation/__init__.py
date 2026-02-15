"""
simulation — Synthetic data generation, stream simulation, and testing tools.

Modules
-------
generate_synthetic_dataset
    Generate a complete synthetic EEG/EMG dataset (NPZ + event TSV files)
    organized by subject/session/trial with a manifest and class map.

simulate_eeg_stream
    Replay a raw CSV file as a simulated real-time EEG data stream,
    delivering packets at configurable speed to a callback.

test_stream_pipeline
    End-to-end test: feed a simulated stream through the
    PreprocessingPipeline and plot raw-vs-filtered results.

compare_filter_configs
    Benchmark multiple filter configurations against ground-truth
    filtered data and visualize the best match.
"""
