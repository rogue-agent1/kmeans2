#!/usr/bin/env python3
"""kmeans2 - K-means, K-means++, and mini-batch K-means clustering."""
import sys, json, math, random

def dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def kmeans_pp_init(X, k, rng):
    centers = [X[rng.randint(0, len(X)-1)]]
    for _ in range(k-1):
        dists = [min(dist(x, c)**2 for c in centers) for x in X]
        total = sum(dists)
        r = rng.random() * total; cumulative = 0
        for i, d in enumerate(dists):
            cumulative += d
            if cumulative >= r: centers.append(X[i]); break
    return centers

def kmeans(X, k, max_iter=100, seed=42):
    rng = random.Random(seed)
    centers = kmeans_pp_init(X, k, rng)
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for x in X:
            c = min(range(k), key=lambda i: dist(x, centers[i]))
            clusters[c].append(x)
        new_centers = []
        for cl in clusters:
            if cl:
                new_centers.append([sum(x[f] for x in cl)/len(cl) for f in range(len(X[0]))])
            else:
                new_centers.append(X[rng.randint(0, len(X)-1)])
        if new_centers == centers: break
        centers = new_centers
    labels = [min(range(k), key=lambda i: dist(x, centers[i])) for x in X]
    inertia = sum(dist(X[i], centers[labels[i]])**2 for i in range(len(X)))
    return centers, labels, inertia

def silhouette(X, labels, k):
    scores = []
    for i in range(len(X)):
        ci = labels[i]
        same = [j for j in range(len(X)) if labels[j] == ci and j != i]
        a = sum(dist(X[i], X[j]) for j in same) / len(same) if same else 0
        b = float('inf')
        for c in range(k):
            if c != ci:
                others = [j for j in range(len(X)) if labels[j] == c]
                if others:
                    b = min(b, sum(dist(X[i], X[j]) for j in others) / len(others))
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        scores.append(s)
    return sum(scores) / len(scores)

def main():
    random.seed(42)
    X = []
    for cx, cy in [(2,2),(8,8),(2,8)]:
        X.extend([[cx+random.gauss(0,0.8), cy+random.gauss(0,0.8)] for _ in range(15)])
    print("K-means clustering demo\n")
    for k in [2, 3, 4]:
        centers, labels, inertia = kmeans(X, k)
        sil = silhouette(X, labels, k)
        print(f"  k={k}: inertia={inertia:.1f}, silhouette={sil:.3f}")
    _, labels3, _ = kmeans(X, 3)
    from collections import Counter
    print(f"\n  k=3 cluster sizes: {dict(Counter(labels3))}")

if __name__ == "__main__":
    main()
