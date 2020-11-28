from utils import *
from statistics import mean

data = prepare_data()

distance_means = []
for k in k_:
    cluster_distances, predicted = clusterize(data, k)

    distances = [c[d] for c, d in zip(cluster_distances[:], predicted)]

    means = mean(distances)
    distance_means.append(means)

plt.plot(k_, distance_means)
plt.xlabel('K')
plt.ylabel('Mean of centroid distance')
plt.grid()
plt.show()


chosen_k = 7

print(f'Te chosen k is: {chosen_k}')

TSNE(data, chosen_k, n_components=2, mine=False, metric='euclidean', print_predicted=True)

TSNE(data, chosen_k, n_components=2, mine=True, metric='euclidean')

TSNE(data, chosen_k, n_components=3, mine=False, metric='euclidean')

TSNE(data, chosen_k, n_components=3, mine=True, metric='euclidean')

