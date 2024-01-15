import torch
import torch.nn as nn
import torch.nn.functional as F

def train_gan(generator, discriminator, data_loader, gen_optimizer, disc_optimizer, loss_function, epochs, device):
    generator.train()
    discriminator.train()

    smooth = 0.1
    noise_factor = 0.05
    gen_train_freq = 1

    for epoch in range(epochs):
        for i, data in enumerate(data_loader):
            batch_size = data.size(0)
            
            real_labels = (1 - smooth) * torch.ones(batch_size, 1).to(device)
            fake_labels = smooth * torch.zeros(batch_size, 1).to(device)

            discriminator.zero_grad()
            real_data = data.to(device).float() + noise_factor * torch.randn(batch_size, *data.shape[1:]).to(device)
            real_output = discriminator(real_data)
            real_loss = loss_function(real_output, real_labels)

            noise = torch.randn(batch_size, generator.input_size).to(device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data.detach())
            fake_loss = loss_function(fake_output, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            disc_optimizer.step()

            if i % gen_train_freq == 0:
                generator.zero_grad()
                fake_data = generator(noise)

                real_features = discriminator.feature_extractor(real_data)
                fake_features = discriminator.feature_extractor(fake_data)

                fm_loss = F.mse_loss(fake_features, real_features.detach())
                g_loss = loss_function(discriminator(fake_data), real_labels)
                total_gen_loss = g_loss + fm_loss
                total_gen_loss.backward()
                gen_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], D Loss: {d_loss.item()}, G Loss: {total_gen_loss.item()}")
