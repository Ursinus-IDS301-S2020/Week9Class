"""
Purpose: To show how to use "Multidimensional Scaling" (MDS)
to find a set of coordinates in 2D that best respect a matrix
"""
from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import glob

def draw_distance_matrix(D, labels, vmax = None):
    """
    Plot a distance matrix with labels in each element
    Parameters
    ----------
    D: narray(N, N)
        A distance matrix
    labels: list (N)
        A list of strings to label each row
    vmax: float
        The max distance to which to scale the plots
        (by default, just the max distance of the matrix,
        but this can be used to make sure colorbars are
        consistent across plots)
    """
    if not vmax:
        vmax = np.max(D)
    N = D.shape[0]
    plt.imshow(D, interpolation='none', cmap='magma_r', vmin=0, vmax=vmax)
    plt.colorbar()
    for i in range(N):
        for j in range(N):
            plt.text(j-0.4, i, "%.2g"%D[i, j], c='white')
    plt.xticks(np.arange(N), labels, rotation=90)
    plt.yticks(np.arange(N), labels)
    plt.ylim([N, -1])


labels = ["Art History", "English", "Math", "CS", "Physics", "Philosophy", "Politics"]

# The "glob" library can be used to list all of the files
# in a directory that match a specified pattern.  In this
# case, we want all of the csv files in the ratings directory
files = glob.glob("Ratings/*.csv")

# Loop through each student's ratings
for f in files:
    # Load in the rating that a particular student gave
    D = np.loadtxt(f, delimiter=',')
    # Use the filename to figure out the student's name
    student = f.split("/")[-1][0:-4]
    # Just in case the student didn't make a symmetric matrix
    # (where Dij = Dji), make it symmetric now by averaging
    # all pairs Dij and Dji
    D = 0.5*(D+D.T)
    
    # Compute multidimensional scaling to find coordinates
    # in 2D that best respect the desired distances
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    X = embedding.fit_transform(D)
    # Compute the distances of the points after MDS
    # so we can compare how close they are to the spec
    DMDS = pairwise_distances(X)

    # Plot the results
    plt.figure(figsize=(16, 5))
    plt.subplot(131)
    draw_distance_matrix(D, labels, vmax=1)
    plt.title("%s's Original Distances"%student)

    plt.subplot(132)
    draw_distance_matrix(DMDS, labels, vmax=1)
    plt.title("MDS distances Distances")

    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 1])
    for i, label in enumerate(labels):
        plt.text(X[i, 0], X[i, 1], label)
    plt.title("MDS Coordinates")
    plt.tight_layout()
    plt.show()