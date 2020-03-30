"""
Purpose: To use the librosa library to "vectorize" audio into features.
The audio is split up into little snippets that are arranged in order
in time.  Each snippet is summarized by 384 dimensions of a "tempogram," 
which is a structure designed to pick up on timbral information 
(e.g. instrumentation / "feel") of the audio
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import librosa

y, sr = librosa.load("MJ.mp3")
hop_length = 512

oenv = librosa.onset.onset_strength(y=y, sr=sr,hop_length=hop_length, max_size=3)
X = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length).T
print(X.shape)
# Compute the times at which each snippet occurs in the audio
times = np.arange(X.shape[0])*hop_length/sr

# Compute the pairwise distances in 384 dimensional space
# between all snippets
D = pairwise_distances(X)
plt.imshow(D, cmap='magma_r', extent=(times[0], times[-1], times[-1], times[0]))
plt.xlabel("Time (Seconds)")
plt.ylabel("Time (Seconds)")
plt.colorbar()
plt.title("Distance matrix for audio snippets")
plt.show()

# Use PCA to reduce the dimension and flatten the 
# data to 2D for visualization.  When we plot it, we
# can see the verse and chorus reside in different
# parts of the space, and it switches between the
# two at around 13 seconds
pca = PCA(n_components=2)
Y = pca.fit_transform(X)

plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], c=times, cmap='magma_r')
plt.colorbar()
plt.title("PCA for Audio Snippets (Colored by Time)")
plt.show()
