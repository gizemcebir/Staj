import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from scipy.io import arff
import matplotlib.pyplot as plt

dosya_yolu = r"C:\Users\gzmce\OneDrive - Firat University\STAJ\artificalGerekli\chainlink.arff"
data, meta = arff.loadarff(dosya_yolu)
df = pd.DataFrame(data)
X = df[['x', 'y', 'z']].values

def natural_neighbor_algorithm(X):
    n = X.shape[0]
    tree = KDTree(X)
    r = 1
    NaN_Edge = set() #doğal komşuları saklar.
    NaN_Num = np.zeros(n, dtype=int) #doğal komşu sayısını sayar

    while True:
        new_edges = set()
        for i in range(n): 
            neighbors_i = tree.query(X[i].reshape(1, -1), r + 1)[1][0] #r. komşu araması
            neighbors_i = neighbors_i[neighbors_i != i] #kendini komşu sayma

            for j in neighbors_i:   # i nin j ye komşusu 
                neighbors_j = tree.query(X[j].reshape(1, -1), r + 1)[1][0]
                neighbors_j = neighbors_j[neighbors_j != j]

                if i in neighbors_j: # j nin i ye komşusu  
                    edge = tuple(sorted((i, j)))
                    if edge not in NaN_Edge and edge not in new_edges:
                        new_edges.add(edge)

        if new_edges:
            for i, j in new_edges:
                NaN_Edge.add((i, j))
                NaN_Num[i] += 1
                NaN_Num[j] += 1  #yeni komşuysa+1 tuple(değişmez) ye ekle
            r += 1
        else:
            break

        if np.all(NaN_Num > 0): #her noktanın komşusu var mı?
            break

    λ = r-1
    return λ, NaN_Edge, NaN_Num

def estimate_nan_eigenvalue(X):
    n = X.shape[0]
    tree = KDTree(X)
    r = 1
    vertexWithoutNeighbor = set(range(n)) #komşusu olmayan noktalar

    while True:  #her noktanın r. komşusu aranır 
        to_remove = set()
        for x in vertexWithoutNeighbor:
            y = tree.query(X[x].reshape(1, -1), r + 1)[1][0]
            y = y[y != x][0]
            y_neighbors = tree.query(X[y].reshape(1, -1), r + 1)[1][0]
            y_neighbors = y_neighbors[y_neighbors != y]
            if x in y_neighbors:  #varsa listeden çıkar
                to_remove.update([x, y])

        if to_remove:  #yoksa stable searching bulundu.
            vertexWithoutNeighbor -= to_remove
            r += 1
        else:
            break

    if vertexWithoutNeighbor:
        weights = []
        for x in vertexWithoutNeighbor: #hala komşusuz yoksa
            neighbors = tree.query(X[x].reshape(1, -1), r + 1)[1][0] #komşu bul
            neighbors = neighbors[neighbors != x]
            min_weight = min(np.linalg.norm(X[neighbors] - X[x], axis=1)) #3D en kısa mesafe
            weights.append(min_weight)#listeye ekle

        weights = sorted(set(weights))
        λ = r
        for i in range(len(weights) - 1):
            if weights[i + 1] - weights[i] <= np.sqrt(weights[i]):
                λ = weights[i]
                break # mesafeler sıralanır ve fark tahmin değeri atanır
    else: #yoksa tur sayısı yani r atanır
        λ = r

    return λ



# 🔹 Algoritmaları çalıştır
λ1, edges, counts = natural_neighbor_algorithm(X)
λ2 = estimate_nan_eigenvalue(X)

# 🔹 Sonuçları yazdır
print(f"Algorithm 1 - Natural Neighbor Eigenvalue (λ): {λ1}")
print(f"Algorithm 2 - Estimated Eigenvalue (λ): {λ2}")
print(f"Toplam doğal komşuluk ilişkisi sayısı: {len(edges)}")
print(f"Komşusu olmayan nokta sayısı: {sum(counts == 0)}")

# 🔹 Görselleştir
import matplotlib.pyplot as plt
plt.show()
