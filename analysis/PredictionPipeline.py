import numpy as np
import torch
import torch.nn as nn

from utils.Model import Model
from torch.utils.data import DataLoader, TensorDataset

class PredictionPipeline:
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2):
        self.model = None

        #Hyperparameters
        self.learning_rate = 0.001
        self.batch_size = 1
        self.epochs = 20 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = nn.MSELoss()

    def run(self, data):
        pass

    # Pass in sequential data ordered by coordination index
    def train_initial_model(self, data: np.ndarray):
        self.model = Model(input_size=data.shape[1], hidden_size=self.hidden_size, num_layers=self.num_layers)

        if isinstance(data, list):
            data = np.stack(data, axis=0)
        seq_len = data.shape[0]
        if seq_len < 2:
            raise ValueError("data must have at least 2 items to form input-output pairs")
            
        n = seq_len - 1  # index of last item
        # inputs: data[i] for i in 0..n-1; targets: data[i+1] for i in 0..n-1
        inputs_np = data[:n]    # data[0], data[1], ..., data[n-1]
        targets_np = data[1:n + 1]  # data[1], data[2], ..., data[n]
        inputs = torch.tensor(inputs_np, dtype=torch.float32)
        targets = torch.tensor(targets_np, dtype=torch.float32)
        dataset = TensorDataset(inputs, targets)
        training_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_inputs, batch_targets in training_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                # Match output/target shapes if necessary
                # If model returns (output, _), unpack
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                # If the model produces only last step, but the target is sequence, align here if needed.
                loss = self.loss_function(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(training_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        torch.save(self.model.state_dict(), "../cache/neural_prediction_model.pth")
        return self.model

    # Train the model on a new data point
    def add_to_model(self, data: np.ndarray):
        if self.model is None:
            raise RuntimeError("model not initialized; call train_initial_model first")
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        # Ensure batch dimension: (n_timesteps, n_features) -> (1, n_timesteps, n_features)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        inputs = torch.tensor(data, dtype=torch.float32).to(self.device)
        targets = torch.tensor(data, dtype=torch.float32).to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        # Align target shape with model output (e.g. model may output (batch, seq, 1))
        if outputs.shape != targets.shape:
            targets = targets[..., : outputs.shape[-1]]
        loss = self.loss_function(outputs, targets)
        loss.backward()
        self.optimizer.step()

    # Predict the next data point
    def predict(self, data: np.ndarray):
        return self.model(data)
