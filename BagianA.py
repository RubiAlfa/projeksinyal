import matplotlib.pyplot as plt
import numpy as np

mu, sigma = 0, 500

x = np.arange(1, 100, 0.1)  # x axis
z = np.random.normal(mu, sigma, len(x))  # noise
y = x ** 2 + z  # data


from scipy.signal import lfilter

n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b, a, y)

fig, ax = plt.subplots(2,1)
ax[0].plot(x, y, linewidth=2, linestyle="-", c="b")
ax[1].plot(x, yy, linewidth=2, linestyle="-", c="b")