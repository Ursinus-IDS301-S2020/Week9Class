# -*- coding: utf-8 -*-
"""
Purpose: To show the basics of creating distance matrices
using the sklearn library's "pairwise_distances" function.
Two clusters of points are created, each with 50 points.
The first cluster is arranged along the first 50 rows
of a matrix, and the second cluster is arranged along
the remaining 50 rows.  This allows us to see a block structure
in the distance matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

X = np.random.randn(100, 2)
X[0:50, :] += [10, 10]

# If you un-comment this line, it will mix the order of
# the points up, and the block structure won't be visible anymore
#X = X[np.random.permutation(100), :]  
plt.scatter(X[:, 0], X[:, 1])


# Creates a matrix which is 100x100
# and the ijth entry holds the distances
# between X[i, :] and X[j, :]
D = pairwise_distances(X)
plt.figure()
plt.imshow(D, cmap='magma_r')
plt.colorbar()