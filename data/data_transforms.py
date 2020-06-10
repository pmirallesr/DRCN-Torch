import numpy as np

class GaussianDenoising:
    """Applies to Tensor. Distort a pixel with 
    additive [+ N(0,scale)] or multiplicative 
    [x N(1,scale)] gaussian noise"""

    def __init__(self, mean = 0, sigma = 0.2, effectType = "additive"):
        means = {"additive": 0, "multiplicative": 1}        
        self.sigma = sigma
        self.effectType = effectType
        if(mean == 0):
            self.mean = means[effectType]

    def __call__(self, x):
        if self.effectType == "multiplicative":
            return x.numpy() * np.random.normal(loc = 1.0, scale = sigma, size = x.shape)
        elif self.effectType == "additive":
            return x.numpy + np.random.normal(loc = 0.0, scale = sigma, size = x.shape)
        else:
            print("Specify a valid type of gaussian error: multiplicative or additive")
            raise ValueError

# Impulse denoising as described in paper
class ImpulseDenoising:
    """Erase a pixel with probability p"""

    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, x):
        return x.numpy() * np.random.binomial(1, self.p, size=x.shape)