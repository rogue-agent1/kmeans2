#!/usr/bin/env python3
"""K-Means clustering. Zero dependencies."""
import math, random

class KMeans:
    def __init__(self, k=3, max_iter=100, seed=42):
        self.k = k; self.max_iter = max_iter; self.seed = seed
        self.centroids = []; self.labels = []

    def _dist(self, a, b):
        return math.sqrt(sum((ai-bi)**2 for ai, bi in zip(a, b)))

    def fit(self, X):
        random.seed(self.seed)
        self.centroids = random.sample(X, min(self.k, len(X)))
        for _ in range(self.max_iter):
            self.labels = [min(range(self.k), key=lambda j: self._dist(x, self.centroids[j])) for x in X]
            new_centroids = []
            for j in range(self.k):
                cluster = [X[i] for i in range(len(X)) if self.labels[i] == j]
                if cluster:
                    d = len(X[0])
                    new_centroids.append([sum(p[dim] for p in cluster)/len(cluster) for dim in range(d)])
                else:
                    new_centroids.append(self.centroids[j])
            if new_centroids == self.centroids: break
            self.centroids = new_centroids
        return self

    def predict(self, X):
        return [min(range(self.k), key=lambda j: self._dist(x, self.centroids[j])) for x in X]

    def inertia(self, X):
        return sum(self._dist(X[i], self.centroids[self.labels[i]])**2 for i in range(len(X)))

if __name__ == "__main__":
    X = [[1,1],[1.5,2],[3,4],[5,7],[3.5,5],[4.5,5]]
    km = KMeans(2).fit(X)
    print(f"Labels: {km.labels}")
