import numpy as np

class KMedoids:
    def __init__(self, n_clusters, max_iters=1000):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
    def fit(self, data):
        n_samples = data.shape[0]
        medoids_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids = data[medoids_idx].copy()
        # print(f"Random initial medoids:\n{self.medoids}")

        distances = self._calc_distances(data)
        
        # Assign each non-medoid data point to the closest medoid
        self.labels = np.argmin(distances, axis=1)
        old_labels = np.empty(n_samples)
        all_idxs = np.arange(n_samples)

        for it in range(self.max_iters):
            best_swap = (-1, -1, 0)
            best_distances = np.zeros(n_samples)
            for i in range(self.n_clusters):
                # Compute the cost of swapping medoid and non-medoid data points
                non_medoids_idx = all_idxs[np.logical_not(np.isin(all_idxs, medoids_idx))]
                for j in non_medoids_idx:
                    new_medoid = data[j]
                    new_distances = np.sum(np.abs(data - new_medoid), axis=1)
                    cost_change = np.sum(new_distances[self.labels == i]) - np.sum(
                        distances[self.labels == i, i]
                    )
                    if cost_change < best_swap[2]:
                        best_swap = (i, j, cost_change)
                        best_distances = new_distances
            if best_swap == (-1, -1, 0):
                break

            i, j, _ = best_swap
            distances[:, i] = best_distances
            self.medoids[i] = data[j]

            self.labels = np.argmin(distances, axis=1)

        # print(f"Number of iterations: {it}")
        # print(f"Final medoids: \n{self.medoids}")

    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)

    def _calc_distances(self, data):
        distances = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sum(np.abs(data - self.medoids[i]), axis=1)
        return distances