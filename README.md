from brisque.brisque_implementation import BrisqueImplementation

# Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11104461.svg)](https://doi.org/10.5281/zenodo.11104461)


BRISQUE is a no-reference image quality score.

A good place to know how BRISQUE works : [LearnOpenCV](https://learnopencv.com/image-quality-assessment-brisque/)


## Installation

```bash
pip install brisque
```

## Usage

1. Trying to perform Image Quality Assessment on **local images** 
```python
from brisque import BRISQUE

obj = BRISQUE()
obj.score("<Ndarray of the Image>")
```

2. Trying to perform Image Quality Assessment on **web images** 
```python
from brisque import BRISQUE

obj = BRISQUE()
obj.score("<URL for the Image>")
```

### Example

#### Local Image

- Input

```python
from brisque import BRISQUE
import numpy as np
from PIL import Image

img_path = "brisque/tests/sample-image.jpg"
img = Image.open(img_path)
ndarray = np.asarray(img)

obj = BRISQUE()
obj.score(image=ndarray)
```
- Output
```
34.84883848208594
```

#### URL

- Input
```python
from brisque import BRISQUE

URL = "https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png"

obj = BRISQUE()
obj.score(URL)
```
- Output
```
71.73427549219988
```

# BRISQUE Numpy Hybrid Pytorch Implementation

This fork features a new pytorch implementation of BRISQUE which has ~1.0X margin error difference compared to the numpy implementation. 
<br>BRISQUE can now be used with either numpy or pytorch.
<br>It can reduce calculations from 30 minutes to 2 minutes based on hardware and similiar.
<br>This allows for enormous faster calculations when using BRISQUE.

Use the following example to get started:
```python
from brisque import BRISQUE
from brisque.brisque_implementation import BrisqueImplementation

obj = BRISQUE(implementation=BrisqueImplementation.Pytorch)
obj.score("https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png")
```