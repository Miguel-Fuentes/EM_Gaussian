import numpy as np
from scipy.stats import norm
from math import sqrt

class gaussian_mix:
    def __init__(self, k):
        self.K = k
    
    def train(self, data, num_iters, fixed_var=False):
        if fixed_var:
            self.params = list(np.random.uniform(low=np.min(data), high=np.max(data), size=self.K))
        else: 
            self.params = list(zip(
                list(np.random.uniform(low=np.min(data), high=np.max(data), size=self.K)), 
                list(np.random.uniform(low=np.std(data, ddof=1)/2, high=np.std(data, ddof=1)*2, size=self.K))
            ))

        self.weights = np.random.uniform(size=self.K)
        self.weights /= np.sum(self.weights)

        for _ in range(num_iters):
            self.probs = []
            if fixed_var:
                for mean in self.params:
                    self.probs.append(norm.pdf(data, loc=mean, scale=1.0))
            else:
                for mean, var in self.params:
                    self.probs.append(norm.pdf(data, loc=mean, scale=sqrt(var)))

            self.probs = np.hstack(self.probs) * self.weights
            self.probs /= np.sum(self.probs, axis=1, keepdims=True)

            n = np.sum(self.probs, axis=0) 
            self.weights = n / data.shape[0]

            means = np.sum(self.probs * data, axis=0) / n
            if fixed_var:
                self.params = list(means)
            else:
                stdevs = np.sum(self.probs * ((np.repeat(data, means.shape[0], axis=1) - means) ** 2), axis=0) / n
                self.params = list(zip(
                    list(means),
                    list(stdevs)
                ))
    def calc_likelihood(self, data):
        pass