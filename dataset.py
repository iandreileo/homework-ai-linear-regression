from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


# Simple dataset
n_samples = 100
n_features = 1
noise = 20
X, y = datasets.make_regression(n_samples=n_samples, n_features=n_features, n_informative=1, noise=noise, random_state=37)
fig = plt.figure(figsize=(4, 4))
plt.xlabel("x (feature)")
plt.ylabel("y (output)")
plt.title("Synthetic data set")
plt.scatter(X, y)
plt.show()

# Complex dataset
n_samples = 300
x = np.linspace(-10, 10, n_samples) # coordinates
noise_sample = np.random.normal(0,0.5,n_samples)
sine_wave = x + np.sin(4*x) + noise_sample
plt.plot(x, sine_wave, 'o')
plt.show()