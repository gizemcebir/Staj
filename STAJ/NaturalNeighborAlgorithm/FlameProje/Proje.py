import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.io import arff
import matplotlib.pyplot as plt

#  Dosya yolunu ayarla
dosya_yolu = r"C:\\Users\\gzmce\\OneDrive - Firat University\\STAJ\\artificalGerekli\\flame.arff"

#  ARFF dosyasını yükle
data, meta = arff.loadarff(dosya_yolu)
df = pd.DataFrame(data)

#  X matrisi (sadece x ve y sütunları), y sınıf etiketleri
X = df[['x', 'y']].values
y = df['class'].astype(int).values
n = X.shape[0]

#  1. Natural Neighbor Algoritması + RKNN

def natural_neighbor_algorithm_with_rknn(X):
    n = X.shape[0]
    tree = KDTree(X)
    r = 1
    NaN_Edge = set()
    NaN_Num = np.zeros(n, dtype=int)

    # Başta boş RKNN listesi
    rknn_list = [[] for _ in range(n)]

    while True:
        new_edges = set()
        for i in range(n):
            neighbors_i = tree.query(X[i].reshape(1, -1), r + 1)[1][0]
            neighbors_i = neighbors_i[neighbors_i != i]

            for j in neighbors_i:
                neighbors_j = tree.query(X[j].reshape(1, -1), r + 1)[1][0]
                neighbors_j = neighbors_j[neighbors_j != j]

                if i in neighbors_j:  # doğal komşuluk
                    edge = tuple(sorted((i, j)))
                    if edge not in NaN_Edge and edge not in new_edges:
                        new_edges.add(edge)

                    # 🔹 RKNN listesine ekle
                    if i not in rknn_list[j]:
                        rknn_list[j].append(i)
                    if j not in rknn_list[i]:
                        rknn_list[i].append(j)

        if new_edges:
            for i, j in new_edges:
                NaN_Edge.add((i, j))
                NaN_Num[i] += 1
                NaN_Num[j] += 1
            r += 1
        else:
            break

        if np.all(NaN_Num > 0):
            break

    λ = r - 1
    return λ, NaN_Edge, NaN_Num, rknn_list


#  2. KNN Kenarları

def knn_edges(X, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nbrs.kneighbors(X)
    edges = set()
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge = tuple(sorted((i, j)))
            edges.add(edge)
    return edges


#  3. Core Points Seçimi

def select_core_points(rknn_list, lam):
    core_points = []
    for i, rknn in enumerate(rknn_list):
        if len(rknn) >= lam:
            core_points.append(i)
    return core_points


#  4. Görselleştirme Fonksiyonları

def plot_edges(X, edges, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30)
    for i, j in edges:
        plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', lw=0.7)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_core_points(X, core_points, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.4, s=30, label="Normal Points")
    plt.scatter(X[core_points, 0], X[core_points, 1], c='red', s=50, label="Core Points")
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


#  5. Çalıştırma

# Natural Neighbor + RKNN

λ1, edges, counts, rknn_list = natural_neighbor_algorithm_with_rknn(X)
print(f"[NaN] Natural Neighbor Eigenvalue (λ): {λ1}")
print(f"[NaN] Toplam doğal komşuluk ilişkisi sayısı: {len(edges)}")
print(f"[NaN] Komşusu olmayan nokta sayısı: {sum(counts == 0)}")

plot_edges(X, edges, "Natural Neighbor Graph")

edges_knn3 = knn_edges(X, 3)
edges_knn5 = knn_edges(X, 5)
plot_edges(X, edges_knn3, "KNN Graph (k=3)")
plot_edges(X, edges_knn5, "KNN Graph (k=5)")

# Core Points
lam = 4
core_points = select_core_points(rknn_list, lam)
print(f"[RKNN] Core Point sayısı: {len(core_points)} / {len(X)}")

plot_core_points(X, core_points, f"RKNN Core Points (λ={lam})")

rknn_sizes = [len(r) for r in rknn_list]
print("RKNN uzunlukları (ilk 20 nokta):", rknn_sizes[:20])
print("Maksimum RKNN uzunluğu:", max(rknn_sizes))


#  6. Core Points üzerinden Clustering

def cluster_core_points_with_borders(core_points, NaN_Edge, X):
    core_set = set(core_points)

    # --- Core pointler için adjacency list ---
    adjacency = {cp: [] for cp in core_points}
    for i, j in NaN_Edge:
        if i in core_set and j in core_set:
            adjacency[i].append(j)
            adjacency[j].append(i)

    # --- DFS ile core clusterları bul ---
    visited = set()
    clusters = []
    core_to_cluster = {}

    for cp in core_points:
        if cp not in visited:
            cluster = []
            stack = [cp]
            visited.add(cp)

            while stack:
                node = stack.pop()
                cluster.append(node)
                core_to_cluster[node] = len(clusters)

                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            clusters.append(cluster)

    # --- Border noktaları en yakın core’a bağla ---
    for i in range(len(X)):
        if i not in core_set:  # normal nokta
            # En yakın core’u bul
            dists = [np.linalg.norm(X[i] - X[cp]) for cp in core_points]
            nearest_core = core_points[np.argmin(dists)]
            cluster_id = core_to_cluster[nearest_core]
            clusters[cluster_id].append(i)

    return clusters


#  7. Cluster Görselleştirme

def plot_clusters(X, clusters, title):
    plt.figure(figsize=(6, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))

    # Her cluster için renk ata
    for idx, cluster in enumerate(clusters):
        plt.scatter(X[cluster, 0], X[cluster, 1], color=colors[idx], s=50, label=f"Cluster {idx+1}")

    plt.scatter(X[:, 0], X[:, 1], color='gray', alpha=0.2, s=20)  # tüm noktalar arka planda
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

#  8. Core Points ile Cluster Oluştur ve Görselleştir

clusters = cluster_core_points_with_borders(core_points, edges, X)
print(f"Toplam Cluster Sayısı: {len(clusters)}")
for idx, cl in enumerate(clusters):
    print(f"Cluster {idx+1} boyutu: {len(cl)}")

plot_clusters(X, clusters, f"Clusters from Core Points (λ={lam})")
