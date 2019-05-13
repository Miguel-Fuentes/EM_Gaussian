import numpy as np
from scipy.stats import norm

FILEPATH = 'em_data.txt'
data = []

with open(FILEPATH) as f:
    for line in f:
        data.append(float(line))
        
data = np.array(data)
data = np.reshape(data, (data.shape[0],1))

k = 3
num_iters = 3

params = list(zip(
    list(np.random.uniform(low=np.min(data), high=np.max(data), size=k)), 
    list(np.random.uniform(high=np.std(data, ddof=1), size=k))
))

weights = np.random.uniform(size=k)
weights /= np.sum(weights)

for _ in range(num_iters):
    probs = []
    for mean, std in params:
        probs.append(norm.pdf(data, loc=mean, scale=std))

    probs = np.hstack(probs)

    preds = probs * weights
    preds /= np.sum(preds, axis=1, keepdims=True)

    n = np.sum(preds, axis=0) 
    weights = n / data.shape[0]

    means = np.sum(preds * data, axis=0) / data.shape[0]
    stdevs = np.sum(preds * (np.repeat(data, means.shape[0], axis=1) - means), axis=0) / data.shape[0]
    params = list(zip(
        list(means),
        list(stdevs)
    ))