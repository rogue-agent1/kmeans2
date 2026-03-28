#!/usr/bin/env python3
"""kmeans2 - K-means clustering from scratch."""
import sys,math,random
def dist(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def kmeans(data,k,max_iter=100):
    centroids=random.sample(data,k)
    for _ in range(max_iter):
        clusters=[[] for _ in range(k)]
        for p in data:
            nearest=min(range(k),key=lambda i:dist(p,centroids[i]));clusters[nearest].append(p)
        new_c=[]
        for cl in clusters:
            if cl:new_c.append([sum(p[j] for p in cl)/len(cl) for j in range(len(cl[0]))])
            else:new_c.append(random.choice(data))
        if new_c==centroids:break
        centroids=new_c
    labels=[]
    for p in data:labels.append(min(range(k),key=lambda i:dist(p,centroids[i])))
    return centroids,labels,clusters
def silhouette(data,labels,k):
    scores=[]
    for i,p in enumerate(data):
        same=[data[j] for j in range(len(data)) if labels[j]==labels[i] and j!=i]
        a=sum(dist(p,q) for q in same)/len(same) if same else 0
        b=float("inf")
        for c in range(k):
            if c!=labels[i]:
                other=[data[j] for j in range(len(data)) if labels[j]==c]
                if other:b=min(b,sum(dist(p,q) for q in other)/len(other))
        scores.append((b-a)/max(a,b) if max(a,b)>0 else 0)
    return sum(scores)/len(scores)
if __name__=="__main__":
    random.seed(42);data=[]
    for cx,cy in[(0,0),(5,5),(10,0)]:data.extend([(cx+random.gauss(0,1),cy+random.gauss(0,1)) for _ in range(30)])
    centroids,labels,clusters=kmeans(data,3)
    for i,c in enumerate(centroids):print(f"Cluster {i}: center=({c[0]:.1f},{c[1]:.1f}), size={len(clusters[i])}")
    print(f"Silhouette score: {silhouette(data,labels,3):.3f}")
