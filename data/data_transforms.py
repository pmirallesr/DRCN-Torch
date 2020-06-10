""" This module contains classes that extend torchvision transforms"""
import numpy as np

class GaussianDenoising(object):
    '''
    Applies to Tensor. Distort a pixel with
        additive [value + N(mean,sigma)] or multiplicative
        [value x N(mean,sigma)] gaussian (normally distributed)
        noise
        Args:
            mean: The mean of the gaussian noise
            sigma: The standard deviation of the gaussian noise
            effect_type (str): multiplicative or additive depending on whether
            you'd like the noise to be added to the pixel value, or the pixel
            value to be multiplied by the noise
    '''

    def __init__(self, mean=0, sigma=0.2, effect_type="additive"):
        means = {"additive": 0, "multiplicative": 1}
        self.sigma = sigma
        self.effect_type = effect_type
        if mean == 0:
            self.mean = means[effect_type]

    def __call__(self, x):
        if self.effect_type == "multiplicative":
            return x.numpy() * np.random.normal(loc=1.0, scale=self.sigma, size=x.shape)
        elif self.effect_type == "additive":
            return x.numpy + np.random.normal(loc=0.0, scale=self.sigma, size=x.shape)
        else:
            print("Specify a valid type of gaussian error: multiplicative or additive")
            raise ValueError


class ImpulseDenoising(object):
    '''
    Erase a pixel with probability p
    Args:
        p: probability of dropping a given pixel
    '''

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return x.numpy() * np.random.binomial(1, self.p, size=x.shape)
