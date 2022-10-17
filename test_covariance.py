import numpy as np
from matplotlib import pyplot as plt

mu,sigma = 0, 0.1

num_samples = np.logspace(1,7,50)
#print(num_samples)

covariances = []

for num in num_samples:

    s1 = np.random.normal(mu, sigma, int(num))
    s2 = np.random.normal(mu, sigma, int(num))

    f1 = s1 + s2
    f2 = s1 - s2

    f = [f1,f2]
    cov = np.cov(f)
    cov_merged = np.concatenate((cov[0],cov[1]),axis=None)
    #print(cov_merged)
    covariances.append(cov_merged)

coefficients = np.array(covariances).T

a11 = coefficients[0]
a12 = coefficients[1]
a21 = coefficients[2]
a22 = coefficients[3]

print(a12)
print(a21)
#

