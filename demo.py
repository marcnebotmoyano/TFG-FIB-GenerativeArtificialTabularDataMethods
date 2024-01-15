import argparse
import torch
import config
from torch.utils.data import DataLoader, random_split
import pandas as pd
from gatdm import VAE
from gatdm import BitDiffusion
from gatdm import train_vae
from gatdm import Generator, Discriminator
from gatdm import train_gan
from gatdm import DataAugmentationModel
from gatdm import train_data_augmentation
from gatdm import train_bit_diffusion
from gatdm import load_data
from gatdm import ks_test
from gatdm import plot_pca, plot_tsne, plot_density_graph

def results_visualization(normalized_real_data_numerical, generated_data_numerical):
    ks_results = ks_test(normalized_real_data_numerical, generated_data_numerical)
    print("Resultados de la Prueba KS:", ks_results)

    plot_density_graph(normalized_real_data_numerical, generated_data_numerical, "Density Plot of Real vs Generated Data")
    plot_pca(normalized_real_data_numerical, generated_data_numerical, "PCA of Real vs Generated Data")
    plot_tsne(normalized_real_data_numerical, generated_data_numerical, "t-SNE of Real vs Generated Data")

def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

def get_num_features(dataset):
    sample_data = dataset.original_data.iloc[:10]
    transformed_sample = dataset.preprocessor.transform(sample_data)
    return transformed_sample.shape[1]

def initialize_and_train_model(model_type, data_loader, validation_loader, device, epochs, input_size, early_stopping_patience=config.EARLY_STOPPING_PATIENCE):
    if model_type == 'gan':
        generator = Generator(input_size=input_size, hidden_dim=config.GAN_HIDDEN_DIM, output_size=input_size).to(device)
        discriminator = Discriminator(input_size=input_size, hidden_dim=config.GAN_HIDDEN_DIM).to(device)
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=config.GAN_LEARNING_RATE, weight_decay=config.GAN_WEIGHT_DECAY)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.GAN_LEARNING_RATE, weight_decay=config.GAN_WEIGHT_DECAY)
        loss_function = torch.nn.BCELoss()

        train_gan(generator, discriminator, data_loader, gen_optimizer, disc_optimizer, loss_function, epochs, device)
        return generator
    
    elif model_type == 'vae':
        hidden_dims = config.VAE_HIDDEN_DIMS
        z_dim = config.VAE_Z_DIM
        vae = VAE(input_size, hidden_dims, z_dim).to(device)
        vae_optimizer = torch.optim.Adam(vae.parameters(), lr=config.LEARNING_RATE)

        train_vae(vae, vae_optimizer, data_loader, validation_loader, epochs, device, config.VAE_INITIAL_BETA, config.VAE_INCREMENT_BETA, config.VAE_MAX_BETA, early_stopping_patience)
        return vae

    elif model_type == 'bitdiff':
        hidden_dim = config.BITDIFF_HIDDEN_DIM
        bitdiff = BitDiffusion(input_size, hidden_dim).to(device)
        bitdiff_optimizer = torch.optim.Adam(bitdiff.parameters(), lr=config.LEARNING_RATE)

        train_bit_diffusion(bitdiff, data_loader, bitdiff_optimizer, epochs, device, max_noise_level=config.MAX_NOISE_LEVEL_BITDIFF)
        return bitdiff
    
    elif model_type == 'data_augmentation':
        hidden_dim = config.DATA_AUGMENTATION_HIDDEN_DIM
        augmentation_model = DataAugmentationModel(input_size, hidden_dim).to(device)

        train_data_augmentation(augmentation_model, data_loader, epochs, device, lr=config.LEARNING_RATE)
        return augmentation_model
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")

def generate_data_with_noise(model, num_samples, device):
    model.eval()
    if isinstance(model, VAE):
        generated_data = model.generate(num_samples, device)
    elif isinstance(model, Generator):
        noise = torch.randn(num_samples, model.input_size).to(device)
        generated_data = model(noise)
    elif isinstance(model, BitDiffusion):
        noise = torch.rand(num_samples, model.input_dim).to(device)
        generated_data = model(noise)
    elif isinstance(model, DataAugmentationModel):
        generated_data = model.generate_data(num_samples, device)
    else:
        raise ValueError("Unsupported model type for data generation.")

    return generated_data.detach().cpu().numpy()

def main(args):
    device = config.DEVICE

    data_loader = load_data(args.dataset_path, batch_size=config.BATCH_SIZE, shuffle=True)

    dataset = data_loader.dataset

    validation_split = config.VALIDATION_SPLIT
    dataset_size = len(dataset)
    validation_size = int(validation_split * dataset_size)
    training_size = dataset_size - validation_size
    training_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])

    training_loader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    early_stopping_patience = config.EARLY_STOPPING_PATIENCE
    num_numerical_columns = len(dataset.get_numerical_columns())
    input_size = get_num_features(dataset)
    model = initialize_and_train_model(args.model_type, training_loader, validation_loader, device, args.epochs, input_size, early_stopping_patience)

    generated_data = generate_data_with_noise(model, args.num_samples, device)
    generated_df = pd.DataFrame(generated_data, columns=dataset.feature_names)
    generated_df.to_csv(config.GENERATED_DATA_DIR, index=False)

    real_data_df = dataset.original_data
    real_data_df.to_csv(config.NORMALIZED_REAL_DATA_DIR, index=False)
    real_data_numerical = real_data_df.select_dtypes(include=['int64', 'float64'])
    real_data_categorical = real_data_df.select_dtypes(exclude=['int64', 'float64'])

    normalized_real_data_numerical = normalize_data(real_data_numerical)

    normalized_real_data = pd.concat([normalized_real_data_numerical, real_data_categorical], axis=1)

    results_visualization(normalized_real_data_numerical.to_numpy(), generated_data[:, :num_numerical_columns])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generación de Datos Tabulares")
    parser.add_argument('--model_type', type=str, required=True, choices=['gan', 'vae', 'bitdiff', 'data_augmentation'], help='Tipo de modelo para generar datos')
    parser.add_argument('--dataset_path', type=str, required=True, help='Ruta al conjunto de datos')
    parser.add_argument('--num_samples', type=int, default=config.NUM_SAMPLES, help='Número de muestras a generar')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Número de épocas para entrenar el modelo')

    args = parser.parse_args()
    main(args)