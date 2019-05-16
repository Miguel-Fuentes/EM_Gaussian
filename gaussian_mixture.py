import numpy as np
from scipy.stats import norm
from math import sqrt

class gaussian_mix:
    def __init__(self, k):
        self.K = k
    
    def train(self, data, num_iters, fixed_var=False, return_likelihood=False):
        if fixed_var:
            self.params = list(np.random.uniform(low=np.min(data), high=np.max(data), size=self.K))
        else: 
            self.params = list(zip(
                list(np.random.uniform(low=np.min(data), high=np.max(data), size=self.K)), 
                list(np.random.uniform(low=np.std(data, ddof=1)/2, high=np.std(data, ddof=1)*2, size=self.K))
            ))

        self.bias = np.random.uniform(size=self.K)
        self.bias /= np.sum(self.bias)
        
        iters = 0
        while iters <= num_iters:
            iters += 1
            weights = []
            if fixed_var:
                for mean in self.params:
                    weights.append(norm.pdf(data, loc=mean, scale=1.0))
            else:
                for mean, var in self.params:
                    weights.append(norm.pdf(data, loc=mean, scale=sqrt(var)))

            if iters == num_iters and return_likelihood:
                probs = np.hstack(weights)
                
            weights = np.hstack(weights) * self.bias
            weights /= np.sum(weights, axis=1, keepdims=True)

            n = np.sum(weights, axis=0) 
            self.bias = n / data.shape[0]

            means = np.sum(weights * data, axis=0) / n
            if fixed_var:
                self.params = list(means)
            else:
                stdevs = np.sum(weights * ((np.repeat(data, means.shape[0], axis=1) - means) ** 2), axis=0) / n
                self.params = list(zip(
                    list(means),
                    list(stdevs)
                ))
                
            if iters == num_iters and return_likelihood:
                weighted_probs = np.sum(weights * probs, axis=1)
                return np.sum(np.log(weighted_probs))
            
if __name__ == '__main__':
    FILEPATH = 'em_data.txt'
    data = []

    with open(FILEPATH) as f:
        for line in f:
            data.append(float(line))

    data = np.array(data)
    data = np.reshape(data, (data.shape[0],1))

    for k in [1, 3, 5]:
        print(f'Mixture of {k} Gaussians')
        model = gaussian_mix(k)
        print('Log Likelihood: ', model.train(data, 200, return_likelihood=True))
        for mean, var in model.params:
            print(f'Mean: {mean}, Variance {var}')