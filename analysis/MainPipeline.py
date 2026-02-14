from PreprocessingPipeline import PreprocessingPipeline
from PredictionPipeline import PredictionPipeline
from PresentPipeline import PresentPipeline

class MainPipeline:
    # window size in seconds
    def __init__(self, window_size_s: float, activation_threshold: float, prediction_data_threshold: float): # 200hz sampling rate
        self.window_size_samples       = int(window_size_s * 200)  # window size in samples   
        self.activation_threshold      = activation_threshold      # neuron activation threshold
        self.prediction_data_threshold = prediction_data_threshold # prediction data threshold
        
        self.data                 = np.array([])
        self.coordination_indexes = np.array([0] * (self.window_size_samples//2))
        self.activation_windows   = np.array([]) # list of activation windows such that (start_index, end_index)

        # Initialize pipelines
        self.preprocessing_pipeline = PreprocessingPipeline() # Stage 1
        self.present_pipeline       = PresentPipeline()       # Stage 2
        self.prediction_pipeline    = PredictionPipeline()    # Stage 3

    def run(self, data):
        if len(self.data) < self.window_size_samples:
            self.data = np.concatenate((self.data, data))
            return
            
        if len(self.activation_windows) > self.prediction_data_threshold:
            self.prediction_pipeline.run(self.data)
        else:
            self.present_pipeline.run(self.data)
            

