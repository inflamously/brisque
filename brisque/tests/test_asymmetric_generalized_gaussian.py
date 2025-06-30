import numpy as np
import skimage
from PIL import Image

from brisque import BRISQUE
from brisque.brisque_asymmetric_generalized_gaussian import BRISQUEAsymmetricGeneralizedGaussianNumpy, \
    BRISQUEAsymmetricGeneralizedGaussianPytorch
from brisque.brisque_pytorch import get_device


def test_asymmetric_generalized_gaussian():
    img_path = "brisque/tests/sample-image.jpg"
    img = Image.open(img_path)
    gray_image = skimage.color.rgb2gray(img)
    brisque = BRISQUE()
    coefficients = brisque.calculate_mscn_coefficients(gray_image)
    coefficients = brisque.calculate_pair_product_coefficients(coefficients)
    for _, coeff in coefficients.items():
        palpha, pmean, psigma_l, psigma_r = BRISQUEAsymmetricGeneralizedGaussianPytorch().fit(coeff, get_device())
        nalpha, nmean, nsigma_l, nsigma_r = BRISQUEAsymmetricGeneralizedGaussianNumpy().fit(coeff.cpu().numpy())
        np.testing.assert_allclose(palpha.cpu().numpy(), nalpha, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(pmean.cpu().numpy(), nmean, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(psigma_l.cpu().numpy(), nsigma_l, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(psigma_r.cpu().numpy(), nsigma_r, rtol=1e-14, atol=1e-14)
