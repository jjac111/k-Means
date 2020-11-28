import os
import xlrd
import pandas as pd
import numpy as np
import seaborn as sns
from config import *
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import manifold
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform

MACHINE_EPSILON = np.finfo(np.double).eps


def prepare_data():
    df = pd.read_excel(raw_data_path)

    input_size = len(df.columns)

    scaler = MinMaxScaler()

    print('Raw dataset:')
    display(df)

    # PREPROCESSING & CLEANING
    num_cols = len(df.columns)
    num_nans = 0
    removed_cols = []
    for col in df.columns:
        if df[col].isnull().values.any():
            num_nans += 1
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
            removed_cols.append(col)
    print(f'\nRemoved {num_cols - len(df.columns)} columns with one unique value: {removed_cols}')
    df.fillna(df.mean(), inplace=True)
    print(f'Found {num_nans} columns with NaN or missing values. Replaced the values with the column mean.\n')
    print('Found duplicated rows:')
    display(df[df.duplicated(keep=False)])
    # df.drop_duplicates(inplace=True)
    df = pd.DataFrame(scaler.fit_transform(df))
    #

    print('\nNormalized dataset:')
    display(df)

    df.to_csv(processed_data_path, header=False, index=False)

    return df


def clusterize(data, k):
    clusterizer = KMeans(k)

    transformed = clusterizer.fit_transform(data)
    predicted = clusterizer.predict(data)

    return transformed, predicted


def TSNE(data, k, n_components=2, mine=True, metric='euclidean', print_predicted=False):
    if mine:
        tsne = myTSNE(n_components=n_components, perplexity=40, n_iter=300, metric=metric)
    else:
        tsne = manifold.TSNE(n_components=n_components, verbose=0, perplexity=40, n_iter=300, metric=metric)

    tsne_results = tsne.fit_transform(data)

    _, predicted = clusterize(data, k)

    data = pd.DataFrame(data)
    data['y'] = pd.Series(predicted)

    if print_predicted:
        print('Predicted clusters:')
        display(data['y'])

    if n_components == 2:
        data['x1'] = tsne_results[:, 0]
        data['x2'] = tsne_results[:, 1]

        plt.figure(figsize=(8, 8))
        sns.scatterplot(
            x="x1", y="x2",
            hue="y",
            palette=sns.color_palette("hls", k),
            data=data,
            legend="full",
            alpha=1
        )

    elif n_components == 3:
        data['x1'] = tsne_results[:, 0]
        data['x2'] = tsne_results[:, 1]
        data['x3'] = tsne_results[:, 2]
        plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter3D(data['x1'], data['x2'], data['x3'], c=data['y'], cmap='CMRmap');

    print('My t-SNE:' if mine else 'sklearn t-SNE:')
    plt.show()


class myTSNE():

    def __init__(self, n_components=2, perplexity=40, n_iter=300, metric='cosine'):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.metric = metric

    ######
    # Code borrowed from Cory Malkin, found at https://towardsdatascience.com/t-sne-python-example-1ded9953f26
    def fit_transform(self, data):
        n_samples = data.shape[0]

        distances = pairwise_distances(data, metric=self.metric)

        P = _joint_probabilities(distances=distances, desired_perplexity=self.perplexity, verbose=False)

        # Reduced feature space
        X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, self.n_components).astype(np.float32)

        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded)

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded):
        params = X_embedded.ravel()

        obj_func = self._kl_divergence

        params = self._gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, self.n_components])

        X_embedded = params.reshape(n_samples, self.n_components)

        return X_embedded

    def _kl_divergence(self, params, P, degrees_of_freedom, n_samples, n_components):
        X_embedded = params.reshape(n_samples, n_components)

        dist = pdist(X_embedded, "sqeuclidean")
        dist /= degrees_of_freedom
        dist += 1.
        dist **= (degrees_of_freedom + 1.0) / -2.0
        Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

        # Kullback-Leibler divergence of P and Q
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

        # Gradient: dC/dY
        grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
        PQd = squareform((P - Q) * dist)
        for i in range(n_samples):
            grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                             X_embedded[i] - X_embedded)
        grad = grad.ravel()
        c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
        grad *= c

        return kl_divergence, grad

    def _gradient_descent(self, obj_func, p0, args, it=0, n_iter_without_progress=300,
                          momentum=0.8, learning_rate=200.0, min_gain=0.01,
                          min_grad_norm=1e-7):

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(np.float).max
        best_error = np.finfo(np.float).max
        best_iter = i = it

        for i in range(it, self.n_iter):

            error, grad = obj_func(p, *args)
            grad_norm = linalg.norm(grad)
            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                break

            if grad_norm <= min_grad_norm:
                break
        return p
    ######
