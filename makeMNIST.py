import numpy as np
import skimage
import os

X = np.loadtxt("mnist_test.csv", delimiter=',')

N = 100
X9 = X[X[:, 0] == 9, 1::]
X1 = X[X[:, 0] == 1, 1::]

for label in np.unique(X[:, 0]):
    folder = "Digits/%i"%label
    if not os.path.exists(folder):
        os.mkdir(folder)
    Xl = X[X[:, 0] == label]
    Xl = Xl[np.random.permutation(Xl.shape[0])[0:N], :]
    for i in range(N):
        x = Xl[i, 1::]
        x = np.reshape(x, (28, 28))
        skimage.io.imsave("%s/%i.png"%(folder, i), x)
