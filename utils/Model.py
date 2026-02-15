import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    """
    LSTM-based sequence-to-sequence model for neural state prediction.

    Supports variable-length sequences via padding and masking.
    Outputs full feature vectors (n_features) at each timestep.
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 32,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            input_size: Number of input features (3 EEG + 1 EMG = 4)
            hidden_size: LSTM hidden dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability between LSTM layers (only if num_layers > 1)
        """
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM: outputs (batch, seq, hidden_size) when batch_first=True
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Project hidden states back to input feature space
        # This allows predicting the next neural state in original space
        self.fc = nn.Linear(hidden_size, input_size)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x, lengths=None, hidden=None):
        """
        Forward pass with optional sequence packing for variable lengths.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
               If padded, lengths must be provided
            lengths: Tensor of actual sequence lengths (batch,) for packing
                     If None, assumes all sequences use full seq_len
            hidden: Optional initial hidden state tuple (h_0, c_0)
                    Each of shape (num_layers, batch, hidden_size)

        Returns:
            output: Predicted sequences of shape (batch, seq_len, input_size)
            hidden: Tuple (h_n, c_n) of final hidden states
        """
        batch_size = x.size(0)

        # Pack sequences if lengths provided (for training with variable lengths)
        if lengths is not None:
            # Sort by length (descending) as required by pack_padded_sequence
            lengths_sorted, perm_idx = lengths.sort(descending=True)
            x_sorted = x[perm_idx]

            # Pack the padded sequence
            packed_input = pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True
            )

            # Pass through LSTM
            packed_output, hidden = self.lstm(packed_input, hidden)

            # Unpack back to padded sequence
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)

            # Restore original order
            _, unperm_idx = perm_idx.sort()
            lstm_out = lstm_out[unperm_idx]

            # Restore hidden state order
            h_n, c_n = hidden
            h_n = h_n[:, unperm_idx, :]
            c_n = c_n[:, unperm_idx, :]
            hidden = (h_n, c_n)
        else:
            # No packing needed - all sequences same length
            lstm_out, hidden = self.lstm(x, hidden)

        # Project to output space: (batch, seq, hidden_size) → (batch, seq, input_size)
        output = self.fc(lstm_out)

        # Apply layer normalization
        output = self.layer_norm(output)

        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden and cell states with zeros.

        Returns:
            Tuple (h_0, c_0) each of shape (num_layers, batch, hidden_size)
        """
        weight = next(self.parameters())
        h_0 = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        return (h_0, c_0)
