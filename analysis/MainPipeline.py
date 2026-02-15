from PreprocessingPipeline import PreprocessingPipeline
from PredictionPipeline import PredictionPipeline
from PresentPipeline import PresentPipeline

from utils.MaxCoordCache import MaxCoordCache

class MainPipeline:
    # window size in seconds
    # TODO find activation threshold
    def __init__(self, window_size_s: float = 0.2, activation_threshold: float = 0.5, prediction_data_threshold: float = 30): # 200hz sampling rate
        self.window_size_samples       = int(window_size_s * 200)  # window size in samples   
        self.activation_threshold      = activation_threshold      # neuron activation threshold
        self.prediction_data_threshold = prediction_data_threshold # prediction data threshold
        self.activation_buffer = 3 # number of points to count before activation
        self.activation_count = 0
        self.temp_activation_window_start = 0
        
        self.data                 = np.array([])
        self.processed_data       = np.array([])
        self.max_n_coord_cache    = MaxNCoordCache(epsilon=0.1) # MaxHeap of each attempt ranked by Coordination Index # Stores attempts

        # Initialize pipelines
        self.preprocessing_pipeline = PreprocessingPipeline() # Stage 1
        self.present_pipeline       = PresentPipeline()       # Stage 2
        self.prediction_pipeline    = PredictionPipeline()    # Stage 3

    # Current_sample is a packet (single time point)
    def run(self, packet):
        # Add packet to data
        self.data = np.concatenate((self.data, packet))
        if len(self.data) < self.window_size_samples:
            return

        # Get current window
        current_window = self.data[len(self.data) - self.window_size_samples:]

        # Preprocess window for downstream processing
        current_sample      = self.preprocessing_pipeline.run(current_window) # note: should be centered around 0
        self.processed_data = np.concatenate((self.processed_data, current_sample[-1])) # only add the newest time point

        # Check for activation
        activation_window = self.check_activation():
        if activation_window: # If activation window is found, get coordination index and add to cache
            coordination_index = self.present_pipeline.get_coordination_index(self.processed_data[activation_window[0]:activation_window[1]])
            self.max_n_coord_cache.add_node(coordination_index, activation_window) # add new attempt to cache
            self.present_pipeline.update_similarity_scores(self.processed_data, activation_window, n_nodes=5) # update top 5 nodes with similarity score
        
        # If enough data has been collected for prediction, run prediction pipeline
        if len(self.activation_windows) >= self.prediction_data_threshold: #if enough data has been collected for prediction
            self.prediction_pipeline.run(current_sample) # TODO STUFF
    
    def check_activation(self):
        if np.math.abs(self.data[-1]) > self.activation_threshold:
            if self.activation_count < self.activation_buffer:
                self.activation_count += 1
                self.temp_activation_window_start = len(self.data) - self.activation_count
            else:
                self.activation_count = 0
                return (self.temp_activation_window_start, len(self.data))
        else:
            self.activation_count = 0
            return None



            

