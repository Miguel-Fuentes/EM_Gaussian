import numpy as np
from scipy.stats import norm

FILEPATH = 'em_data.txt'
data = []

with open(FILEPATH) as f:
    for line in f:
        data.append(float(line))
        
data = np.array(data)
data = np.reshape(data, (data.shape[0],1))

k = 5
num_iters = 100

params = list(zip(
    list(np.random.uniform(low=np.min(data), high=np.max(data), size=k)), 
    list(np.random.uniform(low=np.std(data, ddof=1)/2, high=np.std(data, ddof=1)*2, size=k))
))

weights = np.random.uniform(size=k)
weights /= np.sum(weights)

for _ in range(num_iters):
    probs = []
    for mean, std in params:
        probs.append(norm.pdf(data, loc=mean, scale=std))

    probs = np.hstack(probs) * weights
    probs /= np.sum(probs, axis=1, keepdims=True)

    n = np.sum(probs, axis=0) 
    weights = n / data.shape[0]

    means = np.sum(probs * data, axis=0) / n
    stdevs = np.sum(probs * ((np.repeat(data, means.shape[0], axis=1) - means) ** 2), axis=0) / n
    params = list(zip(
        list(means),
        list(stdevs)
    ))