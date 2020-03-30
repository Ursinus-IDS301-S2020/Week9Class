import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import librosa


wins_per_block=20

y, sr = librosa.load("MJ.mp3")
hop_length = 512

oenv = librosa.onset.onset_strength(y=y, sr=sr,hop_length=hop_length, max_size=3)
X = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length).T


D = pairwise_distances(X)
plt.imshow(D)
plt.show()

pca = PCA(n_components=2)
Y = pca.fit_transform(X)

times = np.arange(Y.shape[0])*hop_length/sr
plt.scatter(Y[:, 0], Y[:, 1], c=times, cmap='magma_r')
plt.colorbar()
plt.show()
