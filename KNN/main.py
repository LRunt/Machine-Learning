import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib.pyplot as plt

#Define f(x)
f = lambda t_: np.sin(2 * np.pi * t_)
#Define f(x) + noise
f_noise = lambda t_: f(t_) + np.random.randn(len(t_))

"""
Drawing the plot
"""
def plot_fun(x=None, y=None, f_est=None):
    t = np.linspace(0, 1, 100)

    plt.figure()
    plt.plot(t, f(t), 'g')
    plt.xlabel('t')
    plt.ylabel('f(t)')

    if x is not None and y is not None:
        plt.scatter(x, y, c='b')
    if f_est is not None:
        plt.plot(t, f_est(t), 'r')

    plt.show()

    """
    Plot aproximation
    N = number of points to aproximate
    M = number of degree    
    """
def plot_approximation(N, M):
    t = np.linspace(0, 1, N)
    y = f_noise(t)

    t2 = np.random.rand(N)
    y2 = f_noise(t2)

    t_t2 = np.concatenate([t, t2])
    y_y2 = np.concatenate([y, y2])
    coeff, stats = P.polyfit(t_t2, y_y2, M, full=True)
    f_est = np.polynomial.Polynomial(coeff)
    plot_fun(t_t2, y_y2, f_est)

if __name__ == '__main__':
    plot_approximation(100, 5)
