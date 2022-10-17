import numpy as np
from matplotlib import pyplot as plt

mu,sigma = 0, 0.1

#array to hold sample numbers
num_samples = np.logspace(1,7,50)

#array to hold calculated covariances
covariances = []

#run a loop, where every cycle the number of samples taken from gaussian distribution is different
for num in num_samples:

    #draw IID gaussian distributed samples s1 and s2
    s1 = np.random.normal(mu, sigma, int(num))
    s2 = np.random.normal(mu, sigma, int(num))

    #define new variables f1,f2 in terms of s1,s2. Change a, b to see effect on covariances
    a=1
    b=1
    f1 = a*s1 + b*s2
    f2 = a*s1 - b*s2

    f = [f1,f2]
    #calculated covariance of f1 and f2
    cov = np.cov(f)
    cov_merged = np.concatenate((cov[0],cov[1]),axis=None)
    #print(cov_merged)
    covariances.append(cov_merged)

coefficients = np.array(covariances).T

samples = np.log(num_samples)

a11 = np.log(np.abs(coefficients[0]))
a12 = np.log(np.abs(coefficients[1]))
a21 = np.log(np.abs(coefficients[2]))
a22 = np.log(np.abs(coefficients[3]))

plt.plot(samples, a11, color='r', label='a11')
plt.plot(samples, a12, color='g', label='a12')
plt.plot(samples, a21, color='b', label='a21')
plt.plot(samples, a22, color='y', label='a22')
  

plt.ylabel("log(aij)")
plt.xlabel("log(# of samples)")
plt.legend()

plt.show()

