#!/usr/bin/env python3
"""kmeans2: K-means clustering with k-means++ initialization."""
import math, random, sys

def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def kmeans_pp_init(X, k):
    centers = [X[random.randint(0, len(X)-1)]]
    for _ in range(k - 1):
        dists = [min(distance(x, c)**2 for c in centers) for x in X]
        total = sum(dists)
        r = random.random() * total
        cumulative = 0
        for i, d in enumerate(dists):
            cumulative += d
            if cumulative >= r:
                centers.append(X[i]); break
    return centers

def kmeans(X, k, max_iter=100, seed=42):
    random.seed(seed)
    centers = kmeans_pp_init(X, k)
    n_features = len(X[0])
    for _ in range(max_iter):
        # Assign
        labels = [min(range(k), key=lambda j: distance(x, centers[j])) for x in X]
        # Update
        new_centers = []
        for j in range(k):
            members = [X[i] for i in range(len(X)) if labels[i] == j]
            if members:
                new_centers.append([sum(m[f] for m in members)/len(members) for f in range(n_features)])
            else:
                new_centers.append(centers[j])
        if new_centers == centers: break
        centers = new_centers
    return labels, centers

def inertia(X, labels, centers):
    return sum(distance(X[i], centers[labels[i]])**2 for i in range(len(X)))

def test():
    X = [[0,0],[0,1],[1,0],[1,1],[10,10],[10,11],[11,10],[11,11]]
    labels, centers = kmeans(X, 2)
    # Should find two clusters
    cluster_0 = {labels[i] for i in range(4)}
    cluster_1 = {labels[i] for i in range(4, 8)}
    assert len(cluster_0) == 1  # All in same cluster
    assert len(cluster_1) == 1
    assert cluster_0 != cluster_1
    # Inertia should be small
    assert inertia(X, labels, centers) < 10
    # k=1
    labels1, centers1 = kmeans(X, 1)
    assert all(l == 0 for l in labels1)
    print("All tests passed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": test()
    else: print("Usage: kmeans2.py test")
