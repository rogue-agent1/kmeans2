from kmeans2 import KMeans
X = [[0,0],[1,0],[0,1],[10,10],[11,10],[10,11]]
km = KMeans(2).fit(X)
assert len(set(km.labels)) == 2
assert km.labels[0] == km.labels[1] == km.labels[2]
assert km.labels[3] == km.labels[4] == km.labels[5]
assert km.labels[0] != km.labels[3]
preds = km.predict([[0.5, 0.5], [10.5, 10.5]])
assert preds[0] != preds[1]
assert km.inertia(X) < 10
print("kmeans2 tests passed")
