import numpy as np
import torch
import torch.nn as nn

from utils.Model import Model
from torch.utils.data import DataLoader, TensorDataset

class PredictionPipeline:
    def __init__(self, hidden_size: int = 32, num_layers: int = 2, input_size: int = 4):
        """
        Initialize the prediction pipeline.

        Args:
            hidden_size: LSTM hidden dimension (default: 32)
            num_layers: Number of LSTM layers (default: 2)
            input_size: Number of input features, 3 EEG + 1 EMG (default: 4)
        """
        self.model = None
        self.optimizer = None

        # Model architecture hyperparameters (FIX: these were missing)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Training hyperparameters
        self.learning_rate = 0.001
        self.batch_size = 1
        self.epochs = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = nn.MSELoss()

    def _pad_sequences(self, sequences: list) -> tuple:
        """
        Pad variable-length sequences to max length in batch.

        Args:
            sequences: List of np.ndarray, each shape (seq_len_i, n_features)

        Returns:
            padded: Tensor of shape (batch, max_seq_len, n_features)
            lengths: Tensor of shape (batch,) containing original lengths
            mask: Boolean tensor of shape (batch, max_seq_len) where True = valid
        """
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        max_len = lengths.max().item()
        batch_size = len(sequences)
        n_features = sequences[0].shape[1]

        # Initialize padded tensor with zeros
        padded = torch.zeros(batch_size, max_len, n_features, dtype=torch.float32)

        # Create mask: True for valid positions, False for padding
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            padded[i, :seq_len, :] = torch.tensor(seq, dtype=torch.float32)
            mask[i, :seq_len] = True

        return padded, lengths, mask

    def _masked_loss(self, predictions, targets, mask):
        """
        Compute MSE loss only on valid (non-padded) positions.

        Args:
            predictions: Tensor (batch, seq, n_features)
            targets: Tensor (batch, seq, n_features)
            mask: Boolean tensor (batch, seq) - True for valid positions

        Returns:
            Scalar loss averaged over valid positions only
        """
        # Expand mask to match feature dimension
        mask_expanded = mask.unsqueeze(-1).expand_as(predictions)

        # Compute squared error
        squared_error = (predictions - targets) ** 2

        # Apply mask and compute mean over valid positions
        masked_error = squared_error * mask_expanded
        loss = masked_error.sum() / mask_expanded.sum()

        return loss

    def run(self, data):
        if not self.model:
            self.train_initial_model(data)
            
        self.add_to_model(data)
        return self.predict(data)

    # Pass in sequential data ordered by coordination index (least→most coordinated)
    def train_initial_model(self, data_list: list, seq_lengths: list):
        """
        Train initial LSTM model on attempts ordered least→most coordinated.

        This ordering guides the model to learn the progression from poor to
        good movements, enabling prediction of improved neural states.

        Args:
            data_list: List of np.ndarray attempts, each shape (seq_len_i, n_features)
                       MUST be sorted in ascending coordination index order
            seq_lengths: List of actual sequence lengths (before padding)
        """
        if len(data_list) < 2:
            raise ValueError("Need at least 2 attempts to train initial model")

        # Initialize model with proper dimensions
        n_features = data_list[0].shape[1]
        self.model = Model(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # Create input→target pairs: predict next attempt from current
        # This teaches the model to predict improvement (current → next better)
        n_pairs = len(data_list) - 1
        inputs_list = data_list[:n_pairs]     # data[0] → data[n-2]
        targets_list = data_list[1:n_pairs+1] # data[1] → data[n-1]

        # Find max length across BOTH inputs and targets for consistent padding
        max_len = max(len(seq) for seq in inputs_list + targets_list)

        # Pad sequences to common max length
        input_lens = torch.tensor([len(seq) for seq in inputs_list], dtype=torch.long)
        target_lens = torch.tensor([len(seq) for seq in targets_list], dtype=torch.long)

        inputs_padded = torch.zeros(n_pairs, max_len, n_features, dtype=torch.float32)
        targets_padded = torch.zeros(n_pairs, max_len, n_features, dtype=torch.float32)
        target_mask = torch.zeros(n_pairs, max_len, dtype=torch.bool)

        for i in range(n_pairs):
            input_len = input_lens[i].item()
            target_len = target_lens[i].item()

            inputs_padded[i, :input_len, :] = torch.tensor(inputs_list[i], dtype=torch.float32)
            targets_padded[i, :target_len, :] = torch.tensor(targets_list[i], dtype=torch.float32)
            target_mask[i, :target_len] = True

        # Move to device
        inputs_padded = inputs_padded.to(self.device)
        targets_padded = targets_padded.to(self.device)
        input_lens = input_lens.to(self.device)
        target_mask = target_mask.to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(inputs_padded, targets_padded, input_lens, target_mask)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        # Note: shuffle=False to maintain coordination ordering

        print(f"Training initial LSTM model on {n_pairs} attempt pairs...")
        print(f"  Input features: {n_features}, Hidden size: {self.hidden_size}")

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            for batch_inputs, batch_targets, batch_lens, batch_mask in dataloader:
                self.optimizer.zero_grad()

                # Forward pass without lengths (process all padded positions)
                # This is necessary because inputs and targets have different lengths
                outputs, _ = self.model(batch_inputs, lengths=None)

                # Compute masked loss (ignore padded positions in targets)
                loss = self._masked_loss(outputs, batch_targets, batch_mask)

                # Backward pass
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        print("Initial model training complete.")

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

    def predict(self, data: np.ndarray, temperature: float = 0.5,
                stochastic: bool = True) -> np.ndarray:
        """
        Predict the next optimal neural state with controlled stochasticity.

        Uses the trained LSTM to predict improved neural patterns. Adds controlled
        Gaussian noise to encourage exploration (stochastic guidance).

        Args:
            data: Current attempt data, shape (seq_len, n_features)
            temperature: Controls stochasticity (default: 0.5)
                        0.0 = deterministic (no noise)
                        0.5 = moderate exploration
                        1.0 = high exploration
            stochastic: Whether to add noise (default: True)

        Returns:
            Predicted optimal neural state, shape (seq_len, n_features)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_initial_model first.")

        self.model.eval()

        with torch.no_grad():
            # Convert to tensor and add batch dimension
            if data.ndim == 2:
                data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            else:
                data_tensor = torch.tensor(data, dtype=torch.float32)

            data_tensor = data_tensor.to(self.device)

            # Forward pass (no lengths needed for single sequence)
            outputs, _ = self.model(data_tensor, lengths=None)

            # Remove batch dimension: (1, seq_len, n_features) → (seq_len, n_features)
            prediction = outputs.squeeze(0)

            # Add stochastic component if requested
            if stochastic and temperature > 0:
                # Estimate noise scale from prediction variance across sequence
                pred_std = prediction.std(dim=0, keepdim=True)

                # Generate Gaussian noise scaled by temperature
                noise = torch.randn_like(prediction) * pred_std * temperature

                # Add noise to prediction
                prediction = prediction + noise

            # Convert back to numpy
            prediction_np = prediction.cpu().numpy()

        return prediction_np
