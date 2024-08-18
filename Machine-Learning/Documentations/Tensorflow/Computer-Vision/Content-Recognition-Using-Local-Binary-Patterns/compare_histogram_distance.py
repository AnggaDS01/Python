from scipy.stats import wasserstein_distance
import numpy as np

def compare_histograms(method, hist1, hist2):
    if method == 'intersection':
        minima = np.minimum(hist1, hist2)
        result = np.sum(minima) / np.sum(hist2)
    elif method == 'chi_square':
        result = 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))  # Avoid division by zero
    elif method == 'euclidean':
        result = np.sqrt(np.sum((hist1 - hist2) ** 2))
    elif method == 'city_block':
        result = np.sum(np.abs(hist1 - hist2))
    elif method == 'bhattacharya':
        result = -np.log(np.sum(np.sqrt(hist1 * hist2)))
    elif method == 'wasserstein':
        result = wasserstein_distance(hist1, hist2)
    else:
        raise ValueError("Method not recognized. Please choose from 'intersection', 'chi_square', 'euclidean', 'city_block', 'bhattacharya', or 'wasserstein'.")
    return result