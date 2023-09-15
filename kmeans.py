class custom_KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        """
        Initialize the KMeans clustering model

        Parameters:
        - n_clusters: int, number of clusters to form.
        - max_iter: int, maximum number of iterations.
        - tol: float, convergence threshold. If the change in centroids is smaller than tol, the algorithm stops.
        - random_state: int or None, seed for random initialization of centroids.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """
        Fit the KMeans model to the input data.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), input data.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize centroids randomly from the data points
        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]

        for iter in range(self.max_iter):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)

            # Update centroids by taking the mean of assigned data points
            new_centroids = self._update_centroids(X, labels)

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

    def _assign_labels(self, X):
        """
        Assign each data point to the nearest centroid.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), input data.

        Returns:
        - labels: numpy array, shape (n_samples,), cluster labels for each data point.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _update_centroids(self, X, labels):
        """
        Update centroids by taking the mean of assigned data points.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), input data.
        - labels: numpy array, shape (n_samples,), cluster labels for each data point.

        Returns:
        - new_centroids: numpy array, shape (n_clusters, n_features), updated centroids.
        """
        new_centroids = np.empty((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If no points are assigned to this centroid, keep it the same
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def predict(self, X):
        """
        Predict cluster labels for input data.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), input data.

        Returns:
        - labels: numpy array, shape (n_samples,), cluster labels for each data point.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def fit_predict(self, X):
        """
        Fit the KMeans model to the input data and immediately predict cluster labels.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), input data.

        Returns:
        - labels: numpy array, shape (n_samples,), cluster labels for each data point.
        """
        self.fit(X)
        return self.predict(X)