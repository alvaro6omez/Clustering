class custom_KMedoids:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        """
        Initialize the KMedoids clustering model.

        Parameters:
        - n_clusters: int, number of clusters to form.
        - max_iter: int, maximum number of iterations.
        - tol: float, convergence threshold. If the change in medoids is smaller than tol, the algorithm stops.
        - random_state: int or None, seed for random initialization of medoids.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.medoids = None

    def fit(self, X):
        """
        Fit the KMedoids model to the input data.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), input data.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, _ = X.shape

        # Initialize medoids randomly from the data points
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids = X[indices]

        for iter in range(self.max_iter):
            # Assign each data point to the nearest medoid
            labels = self._assign_labels(X)

            # Update medoids
            new_medoids = self._update_medoids(X, labels)

            # Check for convergence
            if np.all(np.abs(new_medoids - self.medoids) < self.tol):
                break

            self.medoids = new_medoids

    def _assign_labels(self, X):
        """
        Assign each data point to the nearest medoid.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), input data.

        Returns:
        - labels: numpy array, shape (n_samples,), cluster labels for each data point.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _update_medoids(self, X, labels):
        """
        Update medoids by selecting the data point that minimizes the total dissimilarity within the cluster.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), input data.
        - labels: numpy array, shape (n_samples,), cluster labels for each data point.

        Returns:
        - new_medoids: numpy array, shape (n_clusters, n_features), updated medoids.
        """
        new_medoids = np.empty((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            cluster_distances = np.sum(np.linalg.norm(cluster_points - cluster_points[:, np.newaxis], axis=2), axis=1)
            new_medoids[i] = cluster_points[np.argmin(cluster_distances)]
        return new_medoids

    def predict(self, X):
        """
        Predict cluster labels for input data.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), input data.

        Returns:
        - labels: numpy array, shape (n_samples,), cluster labels for each data point.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def fit_predict(self, X):
        """
        Fit the KMedoids model to the input data and immediately predict cluster labels.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), input data.

        Returns:
        - labels: numpy array, shape (n_samples,), cluster labels for each data point.
        """
        self.fit(X)
        return self.predict(X)
