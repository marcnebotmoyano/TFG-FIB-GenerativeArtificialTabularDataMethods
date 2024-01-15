import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_pca(real_data, generated_data, title="PCA"):
    pca = PCA(n_components=2)
    real_transformed = pca.fit_transform(real_data)
    gen_transformed = pca.transform(generated_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(real_transformed[:, 0], real_transformed[:, 1], c='blue', label='Real Data')
    plt.scatter(gen_transformed[:, 0], gen_transformed[:, 1], c='red', label='Generated Data')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def plot_tsne(real_data, generated_data, title="t-SNE"):
    tsne = TSNE(n_components=2, random_state=42)
    combined_data = np.vstack((real_data, generated_data))
    combined_tsne = tsne.fit_transform(combined_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(combined_tsne[:len(real_data), 0], combined_tsne[:len(real_data), 1], c='blue', label='Real Data')
    plt.scatter(combined_tsne[len(real_data):, 0], combined_tsne[len(real_data):, 1], c='red', label='Generated Data')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_categorical_distributions(real_data, generated_data, categorical_columns):
    for col in categorical_columns:
        fig, ax = plt.subplots()
        real_freq = pd.value_counts(real_data[col])
        gen_freq = pd.value_counts(generated_data[col])
        freq_df = pd.DataFrame({
            "Real": real_freq,
            "Generated": gen_freq
        })
        freq_df.plot(kind='bar', ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    plt.show()

def plot_density_graph(real_data, generated_data, title_prefix="Density Plot"):
    if real_data.shape[1] != generated_data.shape[1]:
        raise ValueError("Real and generated data must have the same number of features")

    num_features = real_data.shape[1]

    # Gráfico de densidad para cada característica individual
    for i in range(num_features):
        plt.figure(figsize=(8, 6))
        sns.kdeplot(real_data[:, i], label='Real Data', color='blue', fill=True)
        sns.kdeplot(generated_data[:, i], label='Generated Data', color='red', fill=True)
        plt.title(f"{title_prefix} - Feature {i+1}")
        plt.xlabel(f"Feature {i+1}")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    # Gráfico de densidad combinado para todas las características
    plt.figure(figsize=(8, 6))
    for i in range(num_features):
        sns.kdeplot(real_data[:, i], label=f'Real Feature {i+1}', fill=True)
        sns.kdeplot(generated_data[:, i], label=f'Generated Feature {i+1}', fill=True)
    plt.title(f"{title_prefix} - All Features")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
