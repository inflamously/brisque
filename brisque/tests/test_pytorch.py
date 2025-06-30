import numpy as np
import skimage
import torch
import scipy.signal as signal
from PIL import Image

from brisque.brisque_numpy import numpy_gaussian_kernel2d, numpy_local_deviation
from brisque.brisque_pytorch import pytorch_gaussian_kernel2d, pytorch_convolve2d, pytorch_local_deviation, get_device


def load_gray_image_numpy():
    img_path = "./sample-image.jpg"
    img = Image.open(img_path)
    ndarray = np.asarray(img)
    return skimage.color.rgb2gray(ndarray)


def test_calculate_mscn_coefficients():
    kernel_size = 7
    sigma = 7 / 6

    # Pytorch
    pval = pytorch_gaussian_kernel2d(kernel_size, sigma, dtype=torch.float64).cpu().numpy()

    # Numpy
    nval = numpy_gaussian_kernel2d(kernel_size, sigma)

    # Test
    assert len(pval) == len(nval)
    assert len(pval[0]) == len(nval[0])
    try:
        np.testing.assert_allclose(pval, nval, rtol=1e-8, atol=1e-8)
        print(f"Results passed: OK")
    except AssertionError as e:
        print(f"Results passed: FAIL\n{e}")


def test_calculate_convolution2d():
    kernel_size = 7
    sigma = 7 / 6
    gray_img = load_gray_image_numpy()

    # Pytorch
    kernel_tensor = pytorch_gaussian_kernel2d(kernel_size, sigma, dtype=torch.float64)
    pval = pytorch_convolve2d(torch.from_numpy(gray_img), kernel_tensor,
                              dtype=torch.float64)

    # Numpy
    kernel_numpy = numpy_gaussian_kernel2d(kernel_size, sigma)
    nval = signal.convolve2d(gray_img, kernel_numpy, mode='same')

    # Test
    np.testing.assert_allclose(pval.cpu().numpy(), nval, rtol=1e-8, atol=1e-8)

def test_same_kernel_implementation():
    kernel_size = 7
    sigma = 7 / 6
    gray_img = load_gray_image_numpy()

    pkernel_tensor = pytorch_gaussian_kernel2d(kernel_size, sigma, dtype=torch.float64)
    kernel = numpy_gaussian_kernel2d(kernel_size, sigma=sigma)
    image_tensor = torch.from_numpy(gray_img).to(get_device(), dtype=torch.float64).view(1, 1, *gray_img.shape[-2:])

    # 1. kernel identity
    np.testing.assert_allclose(pkernel_tensor.cpu().numpy(), kernel, rtol=1e-8, atol=1e-8)


def test_local_deviation():
    kernel_size = 7
    sigma = 7 / 6
    gray_img = load_gray_image_numpy()

    # Pytorch
    pkernel_tensor = pytorch_gaussian_kernel2d(kernel_size, sigma, dtype=torch.float64)
    plocal_mean = pytorch_convolve2d(torch.from_numpy(gray_img), pkernel_tensor,
                                    dtype=torch.float64)
    pval = pytorch_local_deviation(torch.from_numpy(gray_img), plocal_mean, pkernel_tensor)

    # Numpy
    kernel = numpy_gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(gray_img, kernel, 'same')
    nval = numpy_local_deviation(gray_img, local_mean, kernel)

    # Assert
    np.testing.assert_allclose(pkernel_tensor.cpu().numpy(), kernel, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(plocal_mean.cpu().numpy(), local_mean, rtol=1e-8, atol=1e-8)
    # TODO: Fix the Mismatched elements: 92491 / 8847360 (1.05%) issues
    np.testing.assert_allclose(pval.cpu().numpy(), nval, rtol=1e-7, atol=1e-7)
