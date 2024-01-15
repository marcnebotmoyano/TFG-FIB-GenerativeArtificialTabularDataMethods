import torch
from ..models.vae import vae_loss

def train_vae(vae, optimizer, data_loader, validation_loader, epochs, device, initial_beta=1.0, beta_increment=0.1, max_beta=10, early_stopping_patience=10):
    vae.train()
    beta = initial_beta
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_kl, train_recon = 0.0, 0.0, 0.0
        for idx, data in enumerate(data_loader):
            data = data.float().to(device).view(data.size(0), -1)
            optimizer.zero_grad()
            recon_data, mu, log_var = vae(data)
            loss, kl_div, recon_loss = vae_loss(recon_data, data, mu, log_var, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_kl += kl_div.item()
            train_recon += recon_loss.item()

            if idx % 100 == 0:
                print(f"Epoch[{epoch + 1}/{epochs}], Step[{idx}/{len(data_loader)}], Loss: {loss.item()}, KL: {kl_div.item()}, Recon: {recon_loss.item()}, Beta: {beta}")

        # Validación
        if validation_loader:
            vae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_data in validation_loader:
                    val_data = val_data.float().to(device).view(val_data.size(0), -1)
                    recon_data, mu, log_var = vae(val_data)
                    loss, _, _ = vae_loss(recon_data, val_data, mu, log_var, beta)
                    val_loss += loss.item()
            val_loss /= len(validation_loader)
            print(f"Epoch[{epoch + 1}/{epochs}], Validation Loss: {val_loss}")

            # Parada temprana
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        beta = min(beta + beta_increment, max_beta)
        vae.train()

    print("Training completed.")
