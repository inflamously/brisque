from brisque import BRISQUE
import numpy as np
from PIL import Image


def test_validate_url_score():
    URL = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Example.jpg"
    obj = BRISQUE(url=True)
    assert type(round(obj.score(URL), 1)) == type(round(11.11, 1))


def test_validate_local_image():
    img_path = "brisque/tests/sample-image.jpg"
    img = Image.open(img_path)
    ndarray = np.asarray(img)
    obj = BRISQUE(url=False)
    score = obj.score(ndarray)
    print(score)
    assert type(round(score, 1)) == type(round(11.11, 1))


def test_validate_multi_score_image():
    img_path = "brisque/tests/sample-image.jpg"
    img = Image.open(img_path)
    ndarray = np.asarray(img)
    obj = BRISQUE(url=False)
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
    obj = BRISQUE(url=False)
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
    obj = BRISQUE(url=False)
    score = obj.multi_score_images(ndarray, max_workers=4)
    print(score)
    assert type(score) == type([])
    assert type(round(score[0], 1)) == type(round(11.11, 1))
