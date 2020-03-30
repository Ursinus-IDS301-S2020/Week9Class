from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt

D = np.loadtxt("Ratings/Matt.csv", delimiter=',')
D = 0.5*(D+D.T)
categories = ["Art History", "English", "Math", "CS", "Physics", "Philosophy", "Politics"]

embedding = MDS(n_components=2, dissimilarity='precomputed')
X = embedding.fit_transform(D)
plt.scatter(X[:, 0], X[:, 1])
plt.show()