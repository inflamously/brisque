import numpy as np
import scipy.signal as signal


def normalize_kernel(kernel):
    return kernel / np.sum(kernel)


def numpy_gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - int(n / 2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(gaussian_kernel)


def numpy_local_deviation(image, local_mean, kernel):
    "Vectorized approximation of local deviation"
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))
