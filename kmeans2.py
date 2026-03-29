#!/usr/bin/env python3
"""K-means clustering algorithm."""
import sys, random, math

def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def kmeans(points, k, max_iter=100):
    centroids = random.sample(points, k)
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for p in points:
            nearest = min(range(k), key=lambda i: distance(p, centroids[i]))
            clusters[nearest].append(p)
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                d = len(points[0])
                new_centroids.append(tuple(sum(p[j] for p in clusters[i])/len(clusters[i]) for j in range(d)))
            else:
                new_centroids.append(centroids[i])
        if new_centroids == centroids: break
        centroids = new_centroids
    labels = []
    for p in points:
        labels.append(min(range(k), key=lambda i: distance(p, centroids[i])))
    return centroids, labels, clusters

def inertia(points, centroids, labels):
    return sum(distance(points[i], centroids[labels[i]])**2 for i in range(len(points)))

def test():
    random.seed(42)
    cluster1 = [(random.gauss(0,0.3), random.gauss(0,0.3)) for _ in range(20)]
    cluster2 = [(random.gauss(5,0.3), random.gauss(5,0.3)) for _ in range(20)]
    points = cluster1 + cluster2
    centroids, labels, clusters = kmeans(points, 2)
    label_set_1 = set(labels[:20])
    label_set_2 = set(labels[20:])
    assert len(label_set_1) == 1 and len(label_set_2) == 1
    assert label_set_1 != label_set_2
    iner = inertia(points, centroids, labels)
    assert iner < 10
    print("  kmeans2: ALL TESTS PASSED")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": test()
    else: print("K-means clustering")
