import collections
import dataclasses
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import cv2
import numpy as np
import scipy.signal as signal
import torch
from libsvm import svmutil

from brisque import brisque_image
from brisque.brisque_asymmetric_generalized_gaussian import BRISQUEAsymmetricGeneralizedGaussianNumpy, \
    BRISQUEAsymmetricGeneralizedGaussianPytorch
from brisque.brisque_implementation import convert_to_implementation, BrisqueImplementation
from brisque.brisque_numpy import numpy_gaussian_kernel2d, numpy_local_deviation
from brisque.brisque_pytorch import pytorch_gaussian_kernel2d, pytorch_convolve2d, pytorch_local_deviation, get_device
from brisque.models import MODEL_PATH


class BRISQUE:
    def __init__(self, implementation=BrisqueImplementation.Pytorch):
        self.implementation = implementation
        self.model = svmutil.svm_load_model(os.path.join(MODEL_PATH, "svm.txt"))

        with open(os.path.join(MODEL_PATH, "normalize.json")) as f:
            self.scale_params = json.loads(f.read())

    def score(self, image: np.ndarray | str, kernel_size=7, sigma=7 / 6, dtype=torch.float64):
        image = brisque_image.preprocess_img(image)
        image = convert_to_implementation(image, self.implementation, dtype=dtype)

        if self.implementation == BrisqueImplementation.Numpy:
            image_downscaled = cv2.resize(image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_CUBIC)
            kernel = numpy_gaussian_kernel2d(kernel_size, sigma=sigma)
        else:
            new_h, new_w = image.shape[-2] // 2, image.shape[-1] // 2
            image_downscaled = torch.nn.functional.interpolate(
                image.view(1, 1, *image.shape[-2:]),
                size=(new_h, new_w),
                mode="bicubic",
                align_corners=False
            ).squeeze()
            kernel = pytorch_gaussian_kernel2d(kernel_size, sigma=sigma, dtype=dtype)

        brisque_features = self.calculate_brisque_features(image, kernel)
        downscale_brisque_features = self.calculate_brisque_features(image_downscaled, kernel)

        if self.implementation == BrisqueImplementation.Numpy:
            brisque_features = np.concatenate((brisque_features, downscale_brisque_features))
        else:
            brisque_features = torch.concat((brisque_features, downscale_brisque_features))

        return self.calculate_image_quality_score(brisque_features)

    def multi_score(self, image):
        scales = [(5, 5 / 6), (7, 7 / 6), (9, 9 / 6)]
        weights = [0.2, 0.6, 0.2]  # Emphasize standard scale
        scores = []
        for (kernel_size, sigma), weight in zip(scales, weights):
            local_score = self.score(image, kernel_size, sigma)
            scores.append(local_score * weight)
        return sum(scores)

    def score_images(self, images, max_workers=4):
        with ThreadPoolExecutor(max_workers) as executor:
            return list(executor.map(self.score, images))

    def multi_score_images(self, images, max_workers=4):
        with ThreadPoolExecutor(max_workers) as executor:
            return list(executor.map(self.multi_score, images))

    def local_mean(self, image, kernel):
        return signal.convolve2d(image, kernel, 'same')

    def calculate_mscn_coefficients(self, image, kernel, dtype=torch.float64):
        C = 1 / 255

        if self.implementation == BrisqueImplementation.Pytorch:
            local_mean = pytorch_convolve2d(image, kernel, dtype=dtype)
            local_var = pytorch_local_deviation(image, local_mean, kernel, dtype=dtype)
        elif self.implementation == BrisqueImplementation.Numpy:
            local_mean = signal.convolve2d(image, kernel, 'same')
            local_var = numpy_local_deviation(image, local_mean, kernel)
        else:
            return None

        return (image - local_mean) / (local_var + C)

    def calculate_pair_product_coefficients(self, mscn_coefficients):
        return collections.OrderedDict({
            'mscn': mscn_coefficients,
            'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
            'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
            'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
            'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
        })

    def asymmetric_generalized_gaussian_fit(self, x, dtype=torch.float64):
        x = convert_to_implementation(x, self.implementation, dtype=dtype)
        if self.implementation == BrisqueImplementation.Numpy:
            return BRISQUEAsymmetricGeneralizedGaussianNumpy().fit(x)
        elif self.implementation == BrisqueImplementation.Pytorch:
            return BRISQUEAsymmetricGeneralizedGaussianPytorch().fit(x, get_device(), dtype=dtype)
        return None

    def calculate_feature_from_cofficient(self, coefficients_name, coefficients):
        alpha, mean, sigma_l, sigma_r = self.asymmetric_generalized_gaussian_fit(coefficients)

        if coefficients_name == 'mscn':
            features = [alpha, (sigma_l ** 2 + sigma_r ** 2) / 2]
        else:
            features = [alpha, mean, sigma_l ** 2, sigma_r ** 2]

        return features

    def calculate_features(self, coefficients, dtype=torch.float64):
        if self.implementation == BrisqueImplementation.Numpy:
            return [self.calculate_feature_from_cofficient(coefficients_name=name, coefficients=coeff) for name, coeff
                    in coefficients.items()]
        else:
            results = torch.Tensor().to(device=get_device(), dtype=dtype)
            for name, coeff in coefficients.items():
                results = torch.cat([results, torch.stack(
                    self.calculate_feature_from_cofficient(coefficients_name=name, coefficients=coeff))])
            return results

    def calculate_brisque_features(self, image, kernel):
        mscn_coefficients = self.calculate_mscn_coefficients(image, kernel)
        coefficients = self.calculate_pair_product_coefficients(mscn_coefficients)
        features = self.calculate_features(coefficients)
        if self.implementation == BrisqueImplementation.Numpy:
            features = list(chain.from_iterable(features))
            return np.array(features, dtype=object)
        else:
            return features

    def scale_features(self, features, dtype=torch.float64):
        if self.implementation == BrisqueImplementation.Numpy:
            min_ = np.array(self.scale_params['min_'], dtype=object)
            max_ = np.array(self.scale_params['max_'], dtype=object)
        else:
            min_ = torch.Tensor(self.scale_params['min_']).to(device=get_device(), dtype=dtype)
            max_ = torch.Tensor(self.scale_params['max_']).to(device=get_device(), dtype=dtype)

        scale_features = -1 + (2.0 / (max_ - min_) * (features - min_))
        if self.implementation == BrisqueImplementation.Numpy:
            return scale_features
        else:
            return scale_features.cpu().numpy()

    def calculate_image_quality_score(self, brisque_features):
        scaled_brisque_features = self.scale_features(brisque_features)

        x, idx = svmutil.gen_svm_nodearray(
            scaled_brisque_features,
            isKernel=(self.model.param.kernel_type == svmutil.kernel_names.PRECOMPUTED))

        nr_classifier = 1
        prob_estimates = (svmutil.c_double * nr_classifier)()

        return svmutil.libsvm.svm_predict_probability(self.model, x, prob_estimates)
