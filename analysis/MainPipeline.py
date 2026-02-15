from PreprocessingPipeline import PreprocessingPipeline
from PredictionPipeline import PredictionPipeline
from PresentPipeline import PresentPipeline
from LLMPipeline import LLMPipeline
from ProgressMonitor import ProgressMonitor

from utils.MaxCoordCache import MaxNCoordCache, Node

import numpy as np
from collections import deque

class MainPipeline:
    # window size in seconds
    # TODO find activation threshold
    def __init__(
        self,
        window_size_s: float = 0.2,
        activation_threshold: float = 0.2,
        prediction_data_threshold: float = 30,
        api_key: str | None = None,
        movement_class: str = "wrist_flex_ext",
        personality: str = "encouraging",
    ):  # 200hz sampling rate
        self.window_size_samples       = int(window_size_s * 200)  # window size in samples   
        self.activation_threshold      = activation_threshold      # neuron activation threshold
        self.prediction_data_threshold = prediction_data_threshold # prediction data threshold
        self.activation_buffer = 3  # number of points to count before activation
        self.activation_count = 0

        # Sliding window of raw data (only last window_size_samples); deque drops oldest when full
        self.data                 = deque(maxlen=self.window_size_samples)
        # Full history of processed rows (needed for activation_window slicing)
        self.processed_data       = []
        self.max_n_coord_cache    = MaxNCoordCache(epsilon=0.1) # MaxHeap of each attempt ranked by Coordination Index # Stores attempts

        # Initialize pipelines
        self.preprocessing_pipeline = PreprocessingPipeline() # Stage 1
        self.present_pipeline       = PresentPipeline()       # Stage 2
        self.prediction_pipeline    = PredictionPipeline()    # Stage 3

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

        # Check for activation
        activation_window = self.check_activation()
        if activation_window: # If activation window is found, get coordination index and add to cache
            coord_slice = self.processed_data[activation_window[0]:activation_window[1]]
            coordination_index = self.present_pipeline.get_coordination_index(np.array(coord_slice))
            self.max_n_coord_cache.add_node(coordination_index, activation_window) # add new attempt to cache
        
            # If enough data has been collected for prediction, run prediction pipeline
            if len(self.max_n_coord_cache) >= self.prediction_data_threshold: #if enough data has been collected for prediction
                if self.prediction_pipeline.model:
                    self.prediction_pipeline.add_to_model(self.get_data_from_cache(n_nodes=1)[0])
                else:
                    self.prediction_pipeline.train_initial_model(self.get_data_from_cache(n_nodes=len(self.max_n_coord_cache)))

                predicted_ideal_data = self.prediction_pipeline.predict(self.get_data_from_cache(n_nodes=1)[0])
                self.max_n_coord_cache.predicted_ideal_data = predicted_ideal_data # store predicted ideal data for similarity score calculation
                self.max_n_coord_cache.predicted_ideal      = Node(coordination_index=0, activation_window=(0, 0), similarity_score=0)
            
            self.present_pipeline.update_similarity_scores(self.max_n_coord_cache, np.array(self.processed_data), activation_window, n_nodes=5) # update top 5 nodes with similarity score

            # --- Progress monitoring + LLM feedback ---
            # Get similarity score for the current attempt (use the latest node)
            top_nodes = self.max_n_coord_cache.get_top_n_nodes(1)
            current_similarity = top_nodes[0].similarity_score if top_nodes else 0.0

            # Check for plateau / stagnation
            progress_status = self.progress_monitor.add_attempt(coordination_index, current_similarity)

            # Build metrics dict for the LLM
            if self.llm_pipeline is not None:
                top_5 = self.max_n_coord_cache.get_top_n_nodes(5)
                metrics = {
                    "coordination_index": coordination_index,
                    "similarity_score": current_similarity,
                    "attempt_number": len(self.max_n_coord_cache),
                    "coordination_history": list(self.progress_monitor.coordination_history),
                    "movement_class": self._movement_class,
                    "trend": progress_status["trend"],
                    "top_n_scores": [
                        {
                            "coordination_index": n.coordination_index,
                            "similarity_score": n.similarity_score,
                        }
                        for n in top_5
                    ],
                    "has_prediction_model": self.prediction_pipeline.model is not None,
                    "plateau_detected": progress_status["plateau_detected"],
                    "plateau_type": progress_status["plateau_type"],
                    "plateau_details": progress_status["details"],
                }

                self.latest_llm_response = self.llm_pipeline.generate_feedback(metrics)
                
    
    def check_activation(self):
        if np.any(np.abs(self.data[-1]) > self.activation_threshold):
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

    def get_data_from_cache(self, n_nodes: int = 5) -> list:
        nodes = self.max_n_coord_cache.get_top_n_nodes(n_nodes)
        return [
            np.array(self.processed_data[node.activation_window[0]:node.activation_window[1]])
            for node in nodes
        ]

    # ------------------------------------------------------------------
    # LLM convenience methods
    # ------------------------------------------------------------------

    def ask_llm(self, question: str) -> str | None:
        """Forward a user question to the LLM coaching assistant."""
        if self.llm_pipeline is None:
            return None

        # Build current metrics snapshot for context
        top_5 = self.max_n_coord_cache.get_top_n_nodes(5)
        metrics = {
            "coordination_history": list(self.progress_monitor.coordination_history),
            "movement_class": self._movement_class,
            "attempt_number": len(self.max_n_coord_cache),
            "trend": self.progress_monitor.get_trend(),
            "top_n_scores": [
                {
                    "coordination_index": n.coordination_index,
                    "similarity_score": n.similarity_score,
                }
                for n in top_5
            ],
            "has_prediction_model": self.prediction_pipeline.model is not None,
        }
        return self.llm_pipeline.ask(question, current_metrics=metrics)

    def set_personality(self, personality: str) -> None:
        """Switch the LLM coaching personality."""
        if self.llm_pipeline is not None:
            self.llm_pipeline.set_personality(personality)

    def set_movement_class(self, movement_class: str) -> None:
        """Update the current exercise and reset the progress monitor."""
        self._movement_class = movement_class
        self.progress_monitor.reset()
        if self.llm_pipeline is not None:
            self.llm_pipeline.set_movement_class(movement_class)




            

