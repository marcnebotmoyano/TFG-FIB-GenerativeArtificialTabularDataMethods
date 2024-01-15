import torch
from torch import nn
from torch.utils.data import DataLoader

# Función para aplicar ruido a los datos
def apply_noise(data, noise_factor=0.05):
    noise = torch.randn_like(data) * noise_factor
    return data + noise

def random_masking(data, mask_prob=0.1):
    mask = torch.rand_like(data) < mask_prob
    return data.masked_fill(mask, 0)

def train_data_augmentation(model, dataloader, epochs, device, lr=0.001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in dataloader:
            data = data.to(device)

            augmented_data = apply_noise(data)
            augmented_data = random_masking(augmented_data)

            optimizer.zero_grad()

            reconstructed_data = model(augmented_data)

            loss = criterion(reconstructed_data, data)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader)}')
