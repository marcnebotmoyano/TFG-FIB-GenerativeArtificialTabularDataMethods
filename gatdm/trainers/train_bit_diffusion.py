import torch
from torch import nn
import random

def apply_noise(data, noise_factor):
    noise = torch.randn_like(data) * noise_factor
    return data + noise

def train_bit_diffusion(model, dataloader, optimizer, epochs, device, max_noise_level=0.1):
    model.to(device)

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for data in dataloader:
            inputs = data.to(device)

            noise_level = random.uniform(0, max_noise_level)
            noisy_inputs = apply_noise(inputs, noise_level)

            optimizer.zero_grad()

            reconstructed = model(noisy_inputs)

            loss = criterion(reconstructed, inputs)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
