import numpy as np


a = np.load("000001.npy")

with open('a.txt', 'w') as f:
    for i in a:
        f.write(str(i[:9]))
        f.write('\n')

f.close()



