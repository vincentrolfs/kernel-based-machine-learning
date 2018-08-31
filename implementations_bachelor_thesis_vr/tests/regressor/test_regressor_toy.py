import numpy as np
import matplotlib.pyplot as plt

from implementations_bachelor_thesis_vr.Regressor import Regressor

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)

x = np.array([[1], [1.5], [2], [3], [4], [5]])
y = np.array([1, 2, 2, 2.5, 4.5, 5])

r = Regressor(x, y)


def kernel(x, z):
    a = x - z
    return (1 + np.dot(a, a))**(-2) + (1+np.dot(x, z))**2


r.train(kernel, mu=0.15, basis_size=6)

print('coefficients =', r.coefficients)

func = np.vectorize(lambda z: r.predict(z))
func_x = np.linspace(0, 7, 1000)
func_y = func(func_x)

plt.plot(func_x, func_y, '-')
plt.plot(x.reshape(6), y, 'rx')
plt.show()
