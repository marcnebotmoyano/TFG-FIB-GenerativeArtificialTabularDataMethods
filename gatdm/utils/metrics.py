from scipy.stats import ks_2samp, chi2_contingency
import pandas as pd
import numpy as np

def ks_test(real_data, generated_data):
    if real_data.shape[1] != generated_data.shape[1]:
        raise ValueError("Las dimensiones de real_data y generated_data deben coincidir.")

    results = []
    for i in range(real_data.shape[1]):
        stat, p_value = ks_2samp(real_data[:, i], generated_data[:, i])
        results.append((stat, p_value))
    return results