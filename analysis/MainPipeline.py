from analysis.PreprocessingPipeline import PreprocessingPipeline
from analysis.PredictionPipeline import PredictionPipeline
from analysis.PresentPipeline import PresentPipeline

from utils.MaxCoordCache import MaxNCoordCache, Node

import numpy as np
from collections import deque
import threading

class MainPipeline:
    # window size in seconds
    # TODO find activation threshold
    def __init__(self, window_size_s: float = 0.2, activation_threshold: float = 0.2,
                 prediction_data_threshold: float = 30, max_processed_samples: int = 120000,
                 calibration: 'BaselineCalibration' = None, use_whitening: bool = False,
                 adaptive_threshold_n_std: float = 2.5): # 200hz sampling rate
        """
        Args:
            window_size_s: Sliding window size in seconds (default: 0.2s)
            activation_threshold: Voltage threshold for activation detection (default: 0.2)
            prediction_data_threshold: Min attempts before training model (default: 30)
            max_processed_samples: Max samples to keep in processed_data (default: 120000 = 20 min @ 100Hz)
            calibration: Optional BaselineCalibration for adaptive thresholds and artifact rejection
            use_whitening: Apply spatial whitening if calibration provided (default: False)
            adaptive_threshold_n_std: Std multiplier for adaptive thresholds (default: 2.5)
        """
        self.window_size_samples       = int(window_size_s * 200)  # window size in samples
        self.activation_threshold      = activation_threshold      # neuron activation threshold
        self.prediction_data_threshold = prediction_data_threshold # prediction data threshold
        self.max_processed_samples     = max_processed_samples     # memory management threshold
        self.activation_buffer = 3  # number of points to count before activation
        self.activation_count = 0

        # Calibration support
        self.calibration = calibration
        self.use_whitening = use_whitening
        self.adaptive_threshold_n_std = adaptive_threshold_n_std
        self.artifact_count = 0

        # Sliding window of raw data (only last window_size_samples); deque drops oldest when full
        self.data                 = deque(maxlen=self.window_size_samples)
        # Full history of processed rows (needed for activation_window slicing)
        self.processed_data       = []
        self.max_n_coord_cache    = MaxNCoordCache(epsilon=0.1) # MaxHeap of each attempt ranked by Coordination Index # Stores attempts

        # Initialize pipelines
        self.preprocessing_pipeline = PreprocessingPipeline() # Stage 1
        self.present_pipeline       = PresentPipeline()       # Stage 2
        self.prediction_pipeline    = PredictionPipeline(
            hidden_size=32,
            num_layers=2,
            input_size=4
        )    # Stage 3

        # Threading state for async operations
        self.training_in_progress = False
        self.similarity_in_progress = False
        self.prediction_in_progress = False

        # Thread locks for safety
        self.cache_lock = threading.Lock()
        self.model_lock = threading.Lock()

    # Current_sample is a packet (single time point)
    def run(self, packet):
        # Artifact rejection (before any processing)
        if self.calibration and self.calibration.is_calibrated:
            if np.any(self.calibration.is_artifact(packet, n_std=5.0)):
                self.artifact_count += 1
                return  # Skip this packet

            # Optional: apply spatial whitening
            if self.use_whitening:
                packet = self.calibration.apply_whitening(packet.reshape(1, -1)).flatten()

        # Append packet; deque drops oldest when at maxlen (O(1))
        row = np.asarray(packet).reshape(-1)
        self.data.append(row)
        if len(self.data) < self.window_size_samples:
            return

        # Entire buffer is the current window
        current_window = np.array(self.data)

        # Preprocess window for downstream processing
        current_sample      = self.preprocessing_pipeline.run(current_window) # note: should be centered around 0
        # note: ask adnrew to implement run()

        if len(self.processed_data) == 0:
            self.processed_data = [np.asarray(r).reshape(-1) for r in current_sample]
        else:
            self.processed_data.append(np.asarray(current_sample[-1]).reshape(-1))

            # Trim old data if buffer is too large (with buffer to avoid frequent trimming)
            if len(self.processed_data) > self.max_processed_samples + 1000:
                self._trim_processed_data()

        # Check for activation
        activation_window = self.check_activation()
        if activation_window:
            # FAST: Compute coordination index and add to cache (no blocking)
            coord_slice = self.processed_data[activation_window[0]:activation_window[1]]
            coordination_index = self.present_pipeline.get_coordination_index(np.array(coord_slice))

            with self.cache_lock:
                self.max_n_coord_cache.add_node(coordination_index, activation_window)
                cache_size = len(self.max_n_coord_cache)

            # ASYNC: Launch expensive operations in background threads
            if cache_size >= self.prediction_data_threshold:
                # Initial training (VERY EXPENSIVE: ~5-10 seconds)
                if self.prediction_pipeline.model is None and not self.training_in_progress:
                    thread = threading.Thread(
                        target=self._async_train_initial_model,
                        daemon=True,
                        name="InitialTraining"
                    )
                    thread.start()

                # Online learning (MODERATE: ~100ms)
                elif self.prediction_pipeline.model is not None and not self.training_in_progress:
                    thread = threading.Thread(
                        target=self._async_online_learning,
                        daemon=True,
                        name="OnlineLearning"
                    )
                    thread.start()

                # Prediction (FAST but async for consistency)
                if self.prediction_pipeline.model is not None and not self.prediction_in_progress:
                    thread = threading.Thread(
                        target=self._async_predict,
                        daemon=True,
                        name="Prediction"
                    )
                    thread.start()

            # ASYNC: Similarity scoring (MODERATE: ~1ms but 6 comparisons)
            if not self.similarity_in_progress:
                thread = threading.Thread(
                    target=self._async_update_similarity,
                    args=(activation_window,),
                    daemon=True,
                    name="SimilarityScoring"
                )
                thread.start()
                
    
    def check_activation(self):
        current = self.data[-1]

        # Adaptive threshold mode (per-channel thresholds based on calibration)
        if self.calibration and self.calibration.is_calibrated:
            threshold = (self.calibration.channel_mean +
                        self.adaptive_threshold_n_std * self.calibration.channel_std)
            activated = np.any(np.abs(current) > threshold)
        else:
            # Legacy fixed threshold mode
            activated = np.any(np.abs(current) > self.activation_threshold)

        if activated:
            if self.activation_count < self.activation_buffer:
                self.activation_count += 1
            else:
                self.activation_count = 0
                # Indices in processed_data space (last activation_buffer rows)
                if len(self.processed_data) >= self.activation_buffer:
                    return (len(self.processed_data) - self.activation_buffer, len(self.processed_data))
                return None
        else:
            self.activation_count = 0
            return None

    def get_data_from_cache(self, n_nodes: int = 5, reverse_order: bool = False) -> list:
        """
        Retrieve data from cache.

        Args:
            n_nodes: Number of top attempts to retrieve
            reverse_order: If True, return least→most coordinated (for training)

        Returns:
            List of np.ndarray attempts, each shape (seq_len_i, n_features)
        """
        nodes = self.max_n_coord_cache.get_top_n_nodes(n_nodes)

        # Reverse if we want least→most coordinated (for training)
        if reverse_order:
            nodes = nodes[::-1]

        return [
            np.array(self.processed_data[node.activation_window[0]:node.activation_window[1]])
            for node in nodes
        ]

    def _trim_processed_data(self):
        """Trim old data to prevent unbounded memory growth."""
        if len(self.processed_data) <= self.max_processed_samples:
            return

        n_remove = len(self.processed_data) - self.max_processed_samples
        self.processed_data = self.processed_data[n_remove:]

        # Adjust cache activation windows
        with self.cache_lock:
            self.max_n_coord_cache.adjust_windows(-n_remove)
        print(f"Trimmed {n_remove} old samples from buffer")

    # ========== ASYNC OPERATION WRAPPERS ==========

    def _async_train_initial_model(self):
        """
        ASYNC: Train initial LSTM model on all attempts (VERY EXPENSIVE).
        Runs in background thread to avoid blocking data flow.
        """
        self.training_in_progress = True
        try:
            # Snapshot data (thread-safe copy)
            with self.cache_lock:
                n_attempts = len(self.max_n_coord_cache)

            training_data = self.get_data_from_cache(
                n_nodes=n_attempts,
                reverse_order=True  # Least coordinated first
            )
            seq_lengths = [len(seq) for seq in training_data]

            print(f"[ASYNC] Starting initial model training on {n_attempts} attempts...")

            # EXPENSIVE OPERATION (5-10 seconds)
            with self.model_lock:
                self.prediction_pipeline.train_initial_model(training_data, seq_lengths)

            print(f"[ASYNC] Initial model training complete!")

        except Exception as e:
            print(f"[ASYNC ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.training_in_progress = False

    def _async_online_learning(self):
        """
        ASYNC: Fine-tune model on latest best attempt (MODERATE).
        Runs in background thread.
        """
        self.training_in_progress = True
        try:
            # Get latest best attempt
            latest_best = self.get_data_from_cache(n_nodes=1, reverse_order=False)[0]

            # MODERATE OPERATION (~100ms)
            with self.model_lock:
                self.prediction_pipeline.add_to_model(latest_best)

            # print(f"[ASYNC] Online learning update complete")

        except Exception as e:
            print(f"[ASYNC ERROR] Online learning failed: {e}")
        finally:
            self.training_in_progress = False

    def _async_predict(self):
        """
        ASYNC: Predict optimal neural state (FAST but async for consistency).
        Runs in background thread.
        """
        self.prediction_in_progress = True
        try:
            # Get reference attempt
            reference_attempt = self.get_data_from_cache(n_nodes=1, reverse_order=False)[0]

            # FAST OPERATION (~10ms)
            with self.model_lock:
                predicted_ideal_data = self.prediction_pipeline.predict(
                    reference_attempt,
                    temperature=0.5,
                    stochastic=True
                )

            # Update cache with predicted ideal
            with self.cache_lock:
                self.max_n_coord_cache.predicted_ideal_data = predicted_ideal_data
                self.max_n_coord_cache.predicted_ideal = Node(
                    coordination_index=0,
                    activation_window=(0, 0),
                    similarity_score=0
                )

            # print(f"[ASYNC] Prediction complete")

        except Exception as e:
            print(f"[ASYNC ERROR] Prediction failed: {e}")
        finally:
            self.prediction_in_progress = False

    def _async_update_similarity(self, activation_window):
        """
        ASYNC: Update similarity scores for top attempts (MODERATE).
        Runs in background thread.
        """
        self.similarity_in_progress = True
        try:
            # Snapshot data
            processed_data_snapshot = np.array(self.processed_data)

            # MODERATE OPERATION (~5ms with interpolation method)
            with self.cache_lock:
                self.present_pipeline.update_similarity_scores(
                    self.max_n_coord_cache,
                    processed_data_snapshot,
                    activation_window,
                    n_nodes=5
                )

            # print(f"[ASYNC] Similarity scores updated")

        except Exception as e:
            print(f"[ASYNC ERROR] Similarity scoring failed: {e}")
        finally:
            self.similarity_in_progress = False

    # ========== STATUS & MONITORING ==========

    def get_async_status(self) -> dict:
        """
        Get status of all async operations.

        Returns:
            Dictionary with status of training, prediction, and similarity operations
        """
        return {
            'training_in_progress': self.training_in_progress,
            'similarity_in_progress': self.similarity_in_progress,
            'prediction_in_progress': self.prediction_in_progress,
            'model_trained': self.prediction_pipeline.model is not None,
            'num_attempts': len(self.max_n_coord_cache),
            'ready_for_prediction': (
                len(self.max_n_coord_cache) >= self.prediction_data_threshold
            )
        }

    def wait_for_async_operations(self, timeout: float = 30.0):
        """
        Wait for all async operations to complete.
        Useful for testing or graceful shutdown.

        Args:
            timeout: Maximum time to wait in seconds
        """
        import time
        start_time = time.time()

        while (self.training_in_progress or
               self.similarity_in_progress or
               self.prediction_in_progress):

            if time.time() - start_time > timeout:
                print(f"Warning: Timeout waiting for async operations")
                break

            time.sleep(0.1)  # Check every 100ms

    def get_calibration_info(self) -> dict:
        """Get calibration status and artifact statistics.

        Returns:
            Dictionary with calibration status, artifact counts, and threshold info
        """
        if not self.calibration or not self.calibration.is_calibrated:
            return {
                'calibrated': False,
                'mode': 'fixed_threshold',
                'fixed_threshold': self.activation_threshold
            }

        total_packets = len(self.processed_data) + self.artifact_count
        artifact_rate = self.artifact_count / total_packets if total_packets > 0 else 0.0

        # Compute adaptive thresholds
        adaptive_thresholds = (
            self.calibration.channel_mean +
            self.adaptive_threshold_n_std * self.calibration.channel_std
        )

        return {
            'calibrated': True,
            'mode': 'adaptive',
            'adaptive_threshold_n_std': self.adaptive_threshold_n_std,
            'channel_means': self.calibration.channel_mean.tolist(),
            'channel_stds': self.calibration.channel_std.tolist(),
            'adaptive_thresholds': adaptive_thresholds.tolist(),
            'artifacts_rejected': self.artifact_count,
            'total_packets': total_packets,
            'artifact_rate': artifact_rate,
            'whitening_enabled': self.use_whitening
        }






