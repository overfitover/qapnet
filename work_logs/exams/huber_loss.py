import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def huber_loss(x, d):
    x = np.abs(x)
    return ( x<=d ) * x**2 /2 + ( x>d ) * d *( x - d /2.0)

# tf.losses.huber_loss()

if __name__ == '__main__':
    plt.figure(figsize=(6, 4.5), facecolor='w', edgecolor='k')
    x = np.linspace(-20, 20, 10000)
    plt.plot(x, x** 2 / 2, label='squared loss')
    for d in (10, 5, 3, 1):
        plt.plot(x, huber_loss(x, d), label=r'huber loss: $\delta$={}'.format(d))
        plt.legend(loc='best', frameon=False)
        plt.xlabel('residual')
        plt.ylabel('loss')
    plt.show()
