import torch


def get_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def numpy_to_device(nparray, dtype=torch.float64):
    return torch.from_numpy(nparray).to(device=get_device(), dtype=dtype)


def pytorch_gaussian_kernel2d(kernel_size: int, sigma: float,
                              device: torch.device = None, dtype=torch.float64):
    """
    Creates a 2D Gaussian kernel with PyTorch, equivalent to the NumPy version.
    """
    device = get_device(device)
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)

    y, x = torch.meshgrid(
        torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2,
        torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2,
        indexing='ij'
    )
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()        # only one normalisation
    return kernel


def pytorch_convolve2d(image: torch.Tensor, kernel_tensor: torch.Tensor, dtype=torch.float64):
    # 2. The PyTorch Way (GPU/CPU) 🚀
    device = get_device()

    # Convert to tensors and move to the target device
    image_tensor = image.to(device, dtype=dtype).view(1, 1, *image.shape[-2:])

    # Black White Image (W, H) vs. Kernel (7,7) for example.s
    kernel_tensor = kernel_tensor.to(device, dtype=dtype).view(1, 1, *kernel_tensor.shape[-2:])

    # Calculate padding to replicate mode='same'
    padding_size = kernel_tensor.shape[-1] // 2

    # Perform the convolution
    return torch.conv2d(image_tensor, kernel_tensor, padding=padding_size).squeeze().to(device, dtype=dtype)


def pytorch_local_deviation(image: torch.Tensor, local_mean: torch.Tensor, kernel_tensor: torch.Tensor, device=None,
                            dtype=torch.float64):
    "Vectorized approximation of local deviation"
    device = get_device(device)

    image_sigma = image.to(device).view(1, 1, *image.shape[-2:]) ** 2

    # Black White Image (W, H) vs. Kernel (7,7) for example.s
    kernel_tensor = kernel_tensor.to(device, dtype=dtype)
    conv = pytorch_convolve2d(image_sigma, kernel_tensor, dtype=dtype)
    return torch.sqrt(torch.abs(local_mean ** 2 - conv))
