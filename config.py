import torch
# Parámetros Generales de Entrenamiento
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
NUM_SAMPLES = 5000

# GAN
GAN_HIDDEN_DIM = 128
GAN_LEARNING_RATE = 0.0002
GAN_WEIGHT_DECAY = 1e-5

# VAE
VAE_HIDDEN_DIMS = [128, 256]
VAE_Z_DIM = 32
VAE_MAX_BETA = 10
VAE_INITIAL_BETA = 5
VAE_INCREMENT_BETA = 0.1
EARLY_STOPPING_PATIENCE = 10

# BitDiffusion
BITDIFF_HIDDEN_DIM = 128
MAX_NOISE_LEVEL_BITDIFF = 0.1

# Data Augmentation
DATA_AUGMENTATION_HIDDEN_DIM = 128

# Rutas de Archivos y Directorios
DATASET_PATH = 'datasets/original_data/dataset_ovejas.csv'
NORMALIZED_REAL_DATA_DIR = 'datasets/generated_data/normalized_real_data.csv'
GENERATED_DATA_DIR = 'datasets/generated_data/generated_data.csv'

# Configuraciones de Dispositivo
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
