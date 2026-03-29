#!/usr/bin/env python3
"""K-means clustering — Lloyd's algorithm."""
import math, random, sys

class KMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k; self.max_iter = max_iter; self.centroids = []; self.labels = []
    def _dist(self, a, b): return math.sqrt(sum((ai-bi)**2 for ai, bi in zip(a, b)))
    def fit(self, X):
        self.centroids = random.sample(X, self.k)
        for iteration in range(self.max_iter):
            self.labels = [min(range(self.k), key=lambda c: self._dist(x, self.centroids[c])) for x in X]
            new_centroids = []
            for c in range(self.k):
                members = [X[i] for i in range(len(X)) if self.labels[i] == c]
                if members:
                    new_centroids.append([sum(m[d] for m in members)/len(members) for d in range(len(X[0]))])
                else: new_centroids.append(self.centroids[c])
            if new_centroids == self.centroids: break
            self.centroids = new_centroids
        return self.labels
    def inertia(self, X):
        return sum(self._dist(X[i], self.centroids[self.labels[i]])**2 for i in range(len(X)))

if __name__ == "__main__":
    random.seed(42); X = []
    centers = [(0,0), (5,5), (-3,7)]
    for cx, cy in centers:
        for _ in range(30): X.append([cx + random.gauss(0, 1), cy + random.gauss(0, 1)])
    km = KMeans(k=3); labels = km.fit(X)
    from collections import Counter
    print(f"K-Means (k=3, {len(X)} points):")
    for i, c in enumerate(km.centroids):
        count = labels.count(i)
        print(f"  Cluster {i}: center=({c[0]:.2f}, {c[1]:.2f}), size={count}")
    print(f"Inertia: {km.inertia(X):.2f}")
