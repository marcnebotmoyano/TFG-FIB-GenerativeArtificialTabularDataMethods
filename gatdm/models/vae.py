import torch
import torch.nn as nn
import torch.nn.functional as F

def add_regularization(layers, dropout_rate=0.3):
    new_layers = []
    for layer in layers:
        new_layers.append(layer)
        if isinstance(layer, nn.Linear):
            new_layers.append(nn.BatchNorm1d(layer.out_features))
            new_layers.append(nn.Dropout(dropout_rate))
    return new_layers

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super(Encoder, self).__init__()
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU())
            )
            input_dim = h_dim
        self.encoder = nn.Sequential(*add_regularization(modules))
        self.mu = nn.Linear(hidden_dims[-1], z_dim)
        self.log_var = nn.Linear(hidden_dims[-1], z_dim)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.mu(hidden), self.log_var(hidden)

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        hidden_dims.reverse()
        modules = []
        modules.append(nn.Sequential(nn.Linear(z_dim, hidden_dims[0]), nn.ReLU()))
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU())
            )
        modules.append(nn.Sequential(nn.Linear(hidden_dims[-1], output_dim), nn.Sigmoid()))
        self.decoder = nn.Sequential(*add_regularization(modules))

    def forward(self, x):
        return self.decoder(x)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], z_dim=50):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, z_dim)
        self.decoder = Decoder(z_dim, hidden_dims[::-1], input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def generate(self, num_samples, device):
        with torch.no_grad():
            z_dim = self.encoder.mu.out_features
            latent_samples = torch.randn(num_samples, z_dim).to(device)
            generated_data = self.decoder(latent_samples)
        return generated_data

def vae_loss(recon_x, x, mu, log_var, beta=5.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + beta * kl_div
    return total_loss, kl_div, recon_loss
