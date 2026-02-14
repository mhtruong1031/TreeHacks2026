from utils.Model import Model

class PredictionPipeline:
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2):
        self.model = Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        #Hyperparameters
        self.learning_rate = 0.001
        self.batch_size = 1
        self.epochs = 20 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = nn.MSELoss()

    def run(self, data):
        pass

    def train_initial_model(self, data: np.ndarray):
        training_loader = DataLoader(data, batch_size=1, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            for batch in training_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")
        return self.model

    def predict(self, data: np.ndarray):
        return self.model(data)
