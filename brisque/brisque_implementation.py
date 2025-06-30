from typing import Union

import numpy as np
import torch

from brisque.brisque_pytorch import numpy_to_device


class BrisqueImplementation:
    Numpy = "numpy"
    Pytorch = "pytorch"


def convert_to_implementation(data: Union[np.ndarray, torch.Tensor], implementation="pytorch", dtype=torch.float64):
    if implementation == "pytorch":
        if isinstance(data, np.ndarray):
            return numpy_to_device(data, dtype=dtype)
    if implementation == "numpy":
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
    return data
