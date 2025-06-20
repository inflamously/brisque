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
    assert type(round(obj.score(ndarray), 1)) == type(round(11.11, 1))
