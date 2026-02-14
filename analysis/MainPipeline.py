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
        
        self.data                 = np.array([])
        self.max_n_coord_cache    = MaxNCoordCache(epsilon=0.1) # MaxHeap of each attempt ranked by Coordination Index

        # Initialize pipelines
        self.preprocessing_pipeline = PreprocessingPipeline() # Stage 1
        self.present_pipeline       = PresentPipeline()       # Stage 2
        self.prediction_pipeline    = PredictionPipeline()    # Stage 3

    def run(self, current_sample):
        self.data = np.concatenate((self.data, current_sample))
        if len(self.data) < self.window_size_samples:
            return

        current_window = self.data[len(self.data) - self.window_size_samples:]

        current_sample     = self.preprocessing_pipeline.run(current_window)
        coordination_index = self.present_pipeline.get_coordination_index(current_window)

        self.max_n_coord_cache.add_node(coordination_index, (len(self.data) - self.window_size_samples, len(self.data)))
            
        if len(self.activation_windows) >= self.prediction_data_threshold: #if enough data has been collected for prediction
            self.prediction_pipeline.run(current_sample) # TODO
            
            

