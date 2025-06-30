import numpy as np
import scipy.special as special
import scipy.optimize as optimize
import torch


class BRISQUEAsymmetricGeneralizedGaussianNumpy:
    def estimate_phi(self, alpha):
        numerator = special.gamma(2 / alpha) ** 2
        denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
        return numerator / denominator

    def estimate_r_hat(self, x):
        size = np.prod(x.shape)
        return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

    def estimate_R_hat(self, r_hat, gamma):
        numerator = (gamma ** 3 + 1) * (gamma + 1)
        denominator = (gamma ** 2 + 1) ** 2
        return r_hat * numerator / denominator

    def mean_squares_sum(self, x, filter=lambda z: z == z):
        filtered_values = x[filter(x)]
        squares_sum = np.sum(filtered_values ** 2)
        return squares_sum / ((filtered_values.shape))

    def estimate_gamma(self, x):
        left_squares = self.mean_squares_sum(x, lambda z: z < 0)
        right_squares = self.mean_squares_sum(x, lambda z: z >= 0)

        return np.sqrt(left_squares) / np.sqrt(right_squares)

    def estimate_alpha(self, x):
        r_hat = self.estimate_r_hat(x)
        gamma = self.estimate_gamma(x)
        R_hat = self.estimate_R_hat(r_hat, gamma)

        solution = optimize.root(lambda z: self.estimate_phi(z) - R_hat, np.array([0.2])).x

        return solution[0]

    def estimate_sigma(self, x, alpha, filter=lambda z: z < 0):
        return np.sqrt(self.mean_squares_sum(x, filter))

    def estimate_mean(self, alpha, sigma_l, sigma_r, constant):
        return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))

    def fit(self, x):
        alpha = self.estimate_alpha(x)
        sigma_l = self.estimate_sigma(x, alpha, lambda z: z < 0)
        sigma_r = self.estimate_sigma(x, alpha, lambda z: z >= 0)

        constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
        mean = self.estimate_mean(alpha, sigma_l, sigma_r, constant)

        return alpha, mean, sigma_l, sigma_r


class BRISQUEAsymmetricGeneralizedGaussianPytorch:
    def estimate_phi(self, alpha):
        numerator = special.gamma(2 / alpha) ** 2
        denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
        return numerator / denominator

    def estimate_r_hat(self, x):
        size = float(torch.prod(torch.Tensor(list(x.size()))))
        return (torch.sum(torch.abs(x)) / size) ** 2 / (torch.sum(x ** 2) / size)

    def estimate_R_hat(self, r_hat, gamma):
        numerator = (gamma ** 3 + 1) * (gamma + 1)
        denominator = (gamma ** 2 + 1) ** 2
        return r_hat * numerator / denominator

    def mean_squares_sum(self, x, filter=lambda z: z == z):
        filtered_values = x[filter(x)]
        squares_sum = torch.sum(filtered_values ** 2)
        return squares_sum / torch.numel(filtered_values)

    def estimate_gamma(self, x):
        left_squares = self.mean_squares_sum(x, lambda z: z < 0)
        right_squares = self.mean_squares_sum(x, lambda z: z >= 0)

        return torch.sqrt(left_squares) / torch.sqrt(right_squares)

    def estimate_alpha(self, x):
        r_hat = self.estimate_r_hat(x)
        gamma = self.estimate_gamma(x)
        R_hat = self.estimate_R_hat(r_hat, gamma)

        solution = optimize.root(lambda z: self.estimate_phi(z) - R_hat.cpu().numpy(), np.array([0.2])).x

        return solution[0]

    def estimate_sigma(self, x, alpha, filter=lambda z: z < 0):
        return torch.sqrt(self.mean_squares_sum(x, filter))

    def estimate_mean(self, alpha, sigma_l, sigma_r, constant):
        return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))

    def fit(self, x, device, dtype=torch.float64):
        alpha = self.estimate_alpha(x)
        sigma_l = self.estimate_sigma(x, alpha, lambda z: z < 0)
        sigma_r = self.estimate_sigma(x, alpha, lambda z: z >= 0)

        constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
        mean = self.estimate_mean(alpha, sigma_l, sigma_r, constant)

        return torch.from_numpy(np.array(alpha)).to(device, dtype=dtype), mean, sigma_l, sigma_r
