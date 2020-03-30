"""
This shows how to load in a grayscale image of a digit, and
how to flatten it from a 2D array into a 1D array.  In the
1D array, we can think of each location as a dimension.
(We did this back in the steganography assignment)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import skimage
import umap


x = skimage.io.imread("Digits/0/0.png")

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(x, cmap='gray')
plt.colorbar()
print(x.shape)

x = x.flatten()
print(x.shape)
plt.subplot(122)
plt.plot(x)