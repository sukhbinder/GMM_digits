from sklearn.datasets import load_digits
import numpy as np
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


digits = load_digits()


def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)

pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)

# Check the data
plot_digits(digits.data)
plt.show()

gmm = GMM(110, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)

# Generate new data from the model
data_new = gmm.sample(100, random_state=0)
print data_new.shape
digits_new = pca.inverse_transform(data_new)

# Lets display
plot_digits(digits_new)
plt.show()
