import torch
import torch.nn as nn

class DataAugmentationModel(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(DataAugmentationModel, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

    def generate_data(self, num_samples, device):
        random_data = torch.rand(num_samples, self.input_size, device=device)
        with torch.no_grad():
            augmented_data = self.forward(random_data)
        return augmented_data

    