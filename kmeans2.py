#!/usr/bin/env python3
"""kmeans2 - K-means clustering."""
import sys,argparse,json,math,random
def distance(a,b):return math.sqrt(sum((ai-bi)**2 for ai,bi in zip(a,b)))
def kmeans(X,k,max_iter=100):
    centroids=random.sample(X,k)
    for _ in range(max_iter):
        clusters=[[] for _ in range(k)]
        for x in X:
            dists=[distance(x,c) for c in centroids]
            clusters[dists.index(min(dists))].append(x)
        new_centroids=[]
        for cluster in clusters:
            if cluster:new_centroids.append([sum(x[i] for x in cluster)/len(cluster) for i in range(len(cluster[0]))])
            else:new_centroids.append(random.choice(X))
        if new_centroids==centroids:break
        centroids=new_centroids
    labels=[]
    for x in X:
        dists=[distance(x,c) for c in centroids]
        labels.append(dists.index(min(dists)))
    inertia=sum(distance(x,centroids[l])**2 for x,l in zip(X,labels))
    return centroids,labels,inertia
def main():
    p=argparse.ArgumentParser(description="K-means")
    p.add_argument("--k",type=int,default=3);p.add_argument("--samples",type=int,default=150)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    random.seed(args.seed)
    X=[]
    for i in range(args.k):
        cx,cy=random.uniform(-10,10),random.uniform(-10,10)
        for _ in range(args.samples//args.k):X.append([cx+random.gauss(0,1),cy+random.gauss(0,1)])
    centroids,labels,inertia=kmeans(X,args.k)
    from collections import Counter
    cluster_sizes=Counter(labels)
    print(json.dumps({"k":args.k,"samples":len(X),"inertia":round(inertia,2),"centroids":[[round(c,3) for c in cent] for cent in centroids],"cluster_sizes":dict(cluster_sizes)},indent=2))
if __name__=="__main__":main()
