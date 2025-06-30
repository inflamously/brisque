import numpy as np
import torch


def get_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def pytorch_gaussian_kernel2d(kernel_size: int, sigma: float,
                              device: torch.device = None, dtype=torch.float32):
    """
    Creates a 2D Gaussian kernel with PyTorch, equivalent to the NumPy version.
    """
    device = get_device(device)

    # 1. Create a 1D coordinate tensor, matching NumPy's `indices` logic.
    # For an even kernel_size, this creates an asymmetric grid (e.g., -3 to 2 for size 6),
    # which is precisely what `np.indices((n, n)) - int(n / 2)` does.
    grid_indexes = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2

    # 2. Create a 2D grid from the 1D coordinates.
    Y, X = torch.meshgrid(grid_indexes, grid_indexes, indexing='ij')  # Make 6x6

    # 3. Calculate the Gaussian function.
    # We include the `1 / (2 * pi * sigma^2)` term to match the original formula.
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * torch.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    # 4. Normalize the kernel so that its elements sum to 1.
    # This is the crucial step, equivalent to `self.normalize_kernel`, which ensures
    # the kernel acts as a proper weighted average filter.
    return gaussian_kernel / torch.sum(gaussian_kernel)


def pytorch_convolve2d(image: torch.Tensor, kernel_tensor: torch.Tensor, dtype=torch.float32):
    # 2. The PyTorch Way (GPU/CPU) 🚀
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors and move to the target device
    image_tensor = image.to(device, dtype=dtype).view(1, 1, *image.shape[-2:])

    # Black White Image (W, H) vs. Kernel (7,7) for example.s
    kernel_tensor = kernel_tensor.to(device, dtype=dtype).view(1, 1, *kernel_tensor.shape[-2:])

    # Calculate padding to replicate mode='same'
    padding_size = kernel_tensor.shape[-1] // 2

    # Perform the convolution
    return torch.conv2d(image_tensor, kernel_tensor, padding=padding_size).squeeze().to(device, dtype=dtype)


def pytorch_local_deviation(image: torch.Tensor, local_mean: torch.Tensor, kernel_tensor: torch.Tensor, device=None,
                            dtype=torch.float32):
    "Vectorized approximation of local deviation"
    device = get_device(device)

    image_sigma = image ** 2

    # Black White Image (W, H) vs. Kernel (7,7) for example.s
    kernel_tensor = kernel_tensor.to(device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    image_sigma = pytorch_convolve2d(image_sigma, kernel_tensor, dtype=dtype)
    return torch.sqrt(torch.abs(local_mean ** 2 - image_sigma))
