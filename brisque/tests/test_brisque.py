from brisque import BRISQUE
import numpy as np
from PIL import Image
import skimage

from brisque.brisque_implementation import BrisqueImplementation


def test_validate_url_score():
    URL = "https://fastly.picsum.photos/id/10/2500/1667.jpg?hmac=J04WWC_ebchx3WwzbM-Z4_KC_LeLBWr5LZMaAkWkF68"
    obj = BRISQUE()
    score = obj.score(URL)
    print(score)
    assert type(round(score, 1)) == type(round(11.11, 1))


# PYTORCH: PASSED  [100%]34,78944687425334
def test_validate_local_image_pytorch():
    img_path = "brisque/tests/sample-image.jpg"
    img = Image.open(img_path)
    ndarray = np.asarray(img)
    obj = BRISQUE()
    score = obj.score(ndarray)
    print(score)
    assert type(round(score, 1)) == type(round(11.11, 1))


# ORIGNAL: PASSED    [100%]34,84900266778297
def test_validate_local_image_numpy():
    img_path = "brisque/tests/sample-image.jpg"
    img = Image.open(img_path)
    ndarray = np.asarray(img)
    obj = BRISQUE(implementation=BrisqueImplementation.Numpy)
    score = obj.score(ndarray)
    print(score)
    assert type(round(score, 1)) == type(round(11.11, 1))


def test_validate_multi_score_image():
    img_path = "brisque/tests/sample-image.jpg"
    img = Image.open(img_path)
    ndarray = np.asarray(img)
    obj = BRISQUE()
    score = obj.multi_score(ndarray)
    print(score)
    assert type(round(score, 1)) == type(round(11.11, 1))


def test_validate_local_images():
    img_paths = [
        "brisque/tests/sample-image.jpg",
        "brisque/tests/sample-image.jpg",
        "brisque/tests/sample-image.jpg",
        "brisque/tests/sample-image.jpg",
        "brisque/tests/sample-image.jpg",
    ]
    images = [Image.open(img_path) for img_path in img_paths]
    ndarray = [np.asarray(img) for img in images]
    obj = BRISQUE()
    score = obj.score_images(ndarray, max_workers=4)
    print(score)
    assert type(score) == type([])
    assert type(round(score[0], 1)) == type(round(11.11, 1))


def test_validate_multi_score_images():
    img_paths = [
        "brisque/tests/sample-image.jpg",
        "brisque/tests/sample-image.jpg",
        "brisque/tests/sample-image.jpg",
        "brisque/tests/sample-image.jpg",
        "brisque/tests/sample-image.jpg",
    ]
    images = [Image.open(img_path) for img_path in img_paths]
    ndarray = [np.asarray(img) for img in images]
    obj = BRISQUE()
    score = obj.multi_score_images(ndarray, max_workers=4)
    print(score)
    assert type(score) == type([])
    assert type(round(score[0], 1)) == type(round(11.11, 1))


def test_brisque_mscn_coefficients():
    img_path = "brisque/tests/sample-image.jpg"
    img = Image.open(img_path)
    gray_image = skimage.color.rgb2gray(img)
    pval = BRISQUE().calculate_mscn_coefficients(gray_image)
    nval = BRISQUE(implementation=BrisqueImplementation.Numpy).calculate_mscn_coefficients(gray_image)
    np.testing.assert_allclose(pval.cpu().numpy(), nval, rtol=1e-8, atol=1e-8)
