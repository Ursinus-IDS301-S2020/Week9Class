"""
Purpose: To show an example of principal component analysis (PCA)
for doing linear dimension reduction on a collection of image
patches.  Each patch can be thought of as a point in high dimensional
Euclidean space, where each dimension in that space is a pixel, and
the value along that axis is the brightness of that pixel.  
White pixels have 255 as their value, black pixels have 0 as their value,
and gray pixels are somewhere in between.

When you do the dimension reduction with numbers that are very different,
you should see them clustering into two distinct regions.  In this example,
we show that 3s and 0s end up in different parts of the space
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import skimage
from sklearn.decomposition import PCA

def imscatter(X, P, dim, zoom=1):
    """
    Plot patches in specified locations in 2D
    
    Parameters
    ----------
    Y : ndarray (N, 2)
        The positions of each patch in 2D
    P : ndarray (N, dim*dim)
        An array of all of the patches
    dim : int
        The dimension of each patch
    
    """
    #https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
    ax = plt.gca()
    for i in range(P.shape[0]):
        patch = np.reshape(P[i, :], (dim, dim))
        x, y = Y[i, :]
        im = OffsetImage(patch, zoom=zoom, cmap = 'gray')
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)
    ax.update_datalim(X)
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


# Steup an array that will hold 200 images in 784 dimensions (28x28 pixels)
X = np.zeros((200, 784))

# Load in all of the zeros into the first 100 rows
for i in range(100):    
    x = skimage.io.imread("Digits/0/{}.png".format(i))
    x = x.flatten()
    X[i, :] = x # Put image i into row i
    
# Load in all of the threes into the next 100 rows
for i in range(100):    
    x = skimage.io.imread("Digits/3/{}.png".format(i))
    x = x.flatten()
    X[100+i, :] = x # Put image i into row i

# This is an example of what pairwise_distances is actually
# doing to compute distances between each pair of points in X.
# In this case, we're computing Euclidean distance between
# the point at index 10 and the point at index 20
d_10_20 = np.sqrt(np.sum(X[10, :] - X[20, :])**2)

# But it is much faster to use the pairwise_distances function
# to do all of them
# When you run this code, you should see that most of the
# 0s are closer to each other than they are to the 3s, and
# vice versa
D = pairwise_distances(X)
plt.figure()
plt.imshow(D, cmap='magma_r')
plt.xticks([50, 150], ["0s", "3s"])
plt.yticks([50, 150], ["0s", "3s"])
plt.ylim([200, 0])
plt.title("Distance Matrix for Digits")
plt.colorbar()

# When you run this code, you'll see a plot of the images
# at their projected locations
plt.figure(figsize=(12, 12))
pca = PCA(n_components=2)
Y = pca.fit_transform(X)
imscatter(Y, X, 28)
plt.show()

