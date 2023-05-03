
#PCA
from sklearn.manifold import TSNE
#from cuml.manifold import TSNE
from dataset.ClassifierDataset import ClassifierDataset
import matplotlib.pyplot as plt
import numpy as np
dataPrefix="data/patchesIQ_small_shuffled/"
train_dataset = ClassifierDataset(dataPrefix, 'trainMB.h5', 'trainNoMB.h5',num_frames=16)
x_subset = np.zeros((10_000,int(32**2)))
y_subset = np.zeros(10_000)

for ex,i in enumerate(np.random.randint(0, len(train_dataset), 10_000)):
    x_subset[ex] = train_dataset[i][0].flatten().unsqueeze(0).numpy()
    y_subset[ex] = train_dataset[i][1]

#visualising t-SNE again
from sklearn.decomposition import PCA
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x_subset)



tsne = TSNE(random_state = 42, n_components=2,verbose=10, perplexity=150, n_iter=500_000).fit_transform(pca_result_50)

plt.scatter(tsne[:, 0], tsne[:, 1], s= 5, c=y_subset, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(2)-0.5).set_ticks(np.arange(2))
plt.title('Visualizing Kannada MNIST through t-SNE', fontsize=24)
plt.show()
