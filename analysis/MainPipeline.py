from .PreprocessingPipeline import PreprocessingPipeline
from .PredictionPipeline import PredictionPipeline
from .PresentPipeline import PresentPipeline
from .ProgressMonitor import ProgressMonitor
from .LLMPipeline import LLMPipeline

from utils.MaxCoordCache import MaxNCoordCache, Node

import numpy as np
from collections import deque
import threading

class MainPipeline:
    # window size in seconds
    # TODO find activation threshold
    def __init__(
        self,
        window_size_s: float = 0.2,
        activation_threshold: float = 0.2,
        prediction_data_threshold: float = 30,
        max_processed_samples: int = 100_000,
        calibration=None,
        use_whitening: bool = False,
        adaptive_threshold_n_std: float = 2.5,
        api_key: str | None = None,
        movement_class: str = "wrist_flex_ext",
        personality: str = "encouraging",
    ):  # 200hz sampling rate
        self.window_size_samples       = int(window_size_s * 200)  # window size in samples   
        self.activation_threshold      = activation_threshold      # neuron activation threshold
        self.prediction_data_threshold = prediction_data_threshold # prediction data threshold
        self.max_processed_samples     = max_processed_samples     # memory management threshold
        self.activation_buffer = 5  # number of points to count before activation
        self.activation_count = 0
        self.deactivation_count = 0  # number of points below threshold while in window
        self.in_activation_window = False  # whether we're currently in an activation window
        self.activation_start_idx = None  # start index of current window

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

        # Callbacks for immediate updates
        self.on_prediction_ready = None  # Called when prediction completes
        self.on_similarity_ready = None  # Called when similarity scores update

        # Progress monitor (plateau / stagnation detection)
        self.progress_monitor = ProgressMonitor()

        # LLM coaching assistant (optional — degrades gracefully without key)
        self.llm_pipeline = None
        self.latest_llm_response: str | None = None
        self._movement_class = movement_class
        if api_key:
            self.llm_pipeline = LLMPipeline(
                api_key=api_key,
                movement_class=movement_class,
                personality=personality,
            )

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
                self.max_n_coord_cache.add_node(coordination_index, activation_window) # add new attempt to cache

            # If enough data has been collected for prediction, run training/prediction asynchronously
            if len(self.max_n_coord_cache) >= self.prediction_data_threshold:
                # Launch training in background if not already running
                if not self.training_in_progress:
                    self.training_in_progress = True
                    thread = threading.Thread(target=self._async_train_and_predict, daemon=True)
                    thread.start()

            # Launch similarity score update in background if not already running
            if not self.similarity_in_progress:
                self.similarity_in_progress = True
                thread = threading.Thread(
                    target=self._async_update_similarity,
                    args=(activation_window,),
                    daemon=True
                )
                thread.start()

            # ── Plateau / Progress Tracking & LLM Feedback ──────────────
            # Aggregate the similarity scores that update_similarity_scores
            # just wrote onto the top cached nodes.  The mean distance from
            # the current attempt to those nodes is used as the per-attempt
            # similarity value for plateau detection and novelty counting.
            similarity_score = self._aggregate_attempt_similarity(activation_window)

            # Feed into progress monitor → returns plateau / trend status
            progress_status = self.progress_monitor.add_attempt(
                coordination_index=coordination_index,
                similarity_score=similarity_score,
            )

            # If the LLM pipeline is active, build the full metrics dict
            # and (possibly) generate coaching feedback.
            if self.llm_pipeline:
                metrics = self._build_llm_metrics(
                    coordination_index=coordination_index,
                    similarity_score=similarity_score,
                    progress_status=progress_status,
                )
                feedback = self.llm_pipeline.generate_feedback(metrics)
                if feedback:
                    self.latest_llm_response = feedback

    # ------------------------------------------------------------------
    # Async training and prediction methods
    # ------------------------------------------------------------------

    def _async_train_and_predict(self):
        """Train/update model and generate prediction in background thread."""
        try:
            with self.model_lock:
                # Get training data
                if self.prediction_pipeline.model:
                    # Model exists - incremental update with latest attempt
                    data = self.get_data_from_cache(n_nodes=1)[0]
                    self.prediction_pipeline.add_to_model(data)
                else:
                    # Initial training with all collected data
                    all_data = self.get_data_from_cache(n_nodes=len(self.max_n_coord_cache))
                    lengths = [len(x) for x in all_data]
                    self.prediction_pipeline.train_initial_model(all_data, lengths)

                # Generate prediction
                latest_data = self.get_data_from_cache(n_nodes=1)[0]
                predicted_ideal_data = self.prediction_pipeline.predict(latest_data)

                # Calculate similarity score by comparing predicted ideal to top actual attempts
                with self.cache_lock:
                    top_nodes = self.max_n_coord_cache.get_top_n_nodes(5)

                # Calculate average similarity to top attempts
                similarities = []
                for node in top_nodes:
                    actual_attempt = np.array(self.processed_data[node.activation_window[0]:node.activation_window[1]])
                    sim_score = self.present_pipeline.get_similarity_score(predicted_ideal_data, actual_attempt)
                    similarities.append(sim_score)

                # Use mean similarity as the predicted ideal's similarity score
                predicted_similarity = float(np.mean(similarities)) if similarities else 1.0

                # Update cache with predicted ideal (thread-safe)
                with self.cache_lock:
                    self.max_n_coord_cache.predicted_ideal_data = predicted_ideal_data
                    self.max_n_coord_cache.predicted_ideal = Node(
                        coordination_index=0,
                        activation_window=(0, 0),
                        similarity_score=predicted_similarity  # Calculated from actual attempts
                    )

            # Notify GUI that prediction is ready
            if self.on_prediction_ready:
                self.on_prediction_ready()

        except Exception as e:
            print(f"[ERROR] Async training/prediction failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.training_in_progress = False

    def _async_update_similarity(self, activation_window):
        """Update similarity scores for top nodes in background thread."""
        try:
            # Make a copy of processed_data to avoid race conditions
            processed_data_copy = np.array(self.processed_data)

            # Update similarity scores (this does DTW calculations - can be slow)
            self.present_pipeline.update_similarity_scores(
                self.max_n_coord_cache,
                processed_data_copy,
                activation_window,
                n_nodes=5
            )

            # Notify GUI that similarity scores are ready
            if self.on_similarity_ready:
                self.on_similarity_ready()

        except Exception as e:
            print(f"[ERROR] Async similarity update failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.similarity_in_progress = False

    # ------------------------------------------------------------------
    # Plateau / LLM helpers
    # ------------------------------------------------------------------

    def _aggregate_attempt_similarity(self, activation_window) -> float:
        """Mean similarity (distance) of the current attempt to the top
        cached nodes.  Re-uses the scores that
        ``PresentPipeline.update_similarity_scores`` already wrote onto
        each node — no extra computation needed.

        A **high** value means this attempt is far from the best cached
        patterns → novel movement.  A **low** value means it closely
        repeats a known pattern.
        """
        with self.cache_lock:
            top_nodes = self.max_n_coord_cache.get_top_n_nodes(5)
        scores = [
            node.similarity_score
            for node in top_nodes
            if node.activation_window != activation_window
            and node.similarity_score > 0
        ]
        return float(np.mean(scores)) if scores else 0.0

    def _build_llm_metrics(
        self,
        coordination_index: float,
        similarity_score: float,
        progress_status: dict,
    ) -> dict:
        """Assemble the metrics dictionary consumed by ``LLMPipeline``.

        Combines the current attempt's scores, the progress monitor's
        summary (novel-movement counts, coordination progression), and
        the plateau / trend status.
        """
        summary = self.progress_monitor.get_progress_summary()

        with self.cache_lock:
            top_nodes = self.max_n_coord_cache.get_top_n_nodes(5)
        top_n_scores = [
            {
                "coordination_index": n.coordination_index,
                "similarity_score": n.similarity_score,
            }
            for n in top_nodes
        ]

        return {
            # Current attempt
            "coordination_index": coordination_index,
            "similarity_score": similarity_score,
            "attempt_number": summary["total_attempts"],
            "movement_class": self._movement_class,
            "has_prediction_model": self.prediction_pipeline.model is not None,
            # Coordination progression
            "coordination_history": summary["coordination_history_recent"],
            "trend": progress_status["trend"],
            "best_coordination_index": summary["best_coordination_index"],
            "average_coordination_index": summary["average_coordination_index"],
            "coordination_improvement": summary["coordination_improvement"],
            "improvement_rate": summary["improvement_rate"],
            # Novel movement metrics (from similarity scores)
            "novel_movement_count": summary["novel_movement_count"],
            "novel_movement_ratio": summary["novel_movement_ratio"],
            "recent_novel_count": summary["recent_novel_count"],
            "recent_novel_ratio": summary["recent_novel_ratio"],
            # Plateau / stagnation
            "plateau_detected": progress_status["plateau_detected"],
            "plateau_type": progress_status["plateau_type"],
            "plateau_details": progress_status["details"],
            # Top cached attempts
            "top_n_scores": top_n_scores,
        }

    def check_activation(self):
        """
        Check for activation with buffers at both start and end.

        Start buffer: requires activation_buffer consecutive high points to START window
        End buffer: requires activation_buffer consecutive low points to END window

        Returns:
            tuple (start_idx, end_idx) if window just closed, None otherwise
        """
        current = self.data[-1]

        # Adaptive threshold mode (per-channel thresholds based on calibration)
        if self.calibration and self.calibration.is_calibrated:
            threshold = (self.calibration.channel_mean +
                        self.adaptive_threshold_n_std * self.calibration.channel_std)
            activated = np.any(np.abs(current) > threshold)
        else:
            # Legacy fixed threshold mode
            activated = np.any(np.abs(current) > self.activation_threshold)

        if not self.in_activation_window:
            # Not in window - check for START condition
            if activated:
                self.activation_count += 1
                if self.activation_count >= self.activation_buffer:
                    # Start buffer filled - begin activation window
                    self.in_activation_window = True
                    self.activation_start_idx = len(self.processed_data) - self.activation_buffer
                    self.activation_count = 0
                    self.deactivation_count = 0
                    print(f"[DEBUG] Activation window START at index {self.activation_start_idx}")
            else:
                # Reset if signal drops before reaching buffer
                self.activation_count = 0
            return None

        else:
            # In window - check for END condition
            if activated:
                # Signal still high - reset deactivation counter and continue window
                self.deactivation_count = 0
                return None
            else:
                # Signal dropped - count towards end buffer
                self.deactivation_count += 1
                if self.deactivation_count >= self.activation_buffer:
                    # End buffer filled - close activation window
                    end_idx = len(self.processed_data) - self.activation_buffer
                    window = (self.activation_start_idx, end_idx)

                    # Reset state
                    self.in_activation_window = False
                    self.activation_start_idx = None
                    self.deactivation_count = 0
                    self.activation_count = 0

                    print(f"[DEBUG] Activation window END: {window[0]} -> {window[1]} ({window[1]-window[0]} samples)")

                    # Only return valid windows (must have at least activation_buffer samples)
                    if window[1] > window[0]:
                        return window

                return None

    def get_data_from_cache(self, n_nodes: int = 5, reverse_order: bool = False) -> list:
        """
        Retrieve data from cache (thread-safe).

        Args:
            n_nodes: Number of top attempts to retrieve
            reverse_order: If True, return least→most coordinated (for training)

        Returns:
            List of np.ndarray attempts, each shape (seq_len_i, n_features)
        """
        with self.cache_lock:
            nodes = self.max_n_coord_cache.get_top_n_nodes(n_nodes)

        # Reverse if we want least→most coordinated (for training)
        if reverse_order:
            nodes = nodes[::-1]

        return [
            np.array(self.processed_data[node.activation_window[0]:node.activation_window[1]])
            for node in nodes
        ]






