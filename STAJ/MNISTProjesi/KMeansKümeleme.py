from sklearn.cluster import KMeans
import numpy as np
from Proje import load_mnist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


X_train, y_train, X_test, y_test = load_mnist()


# 10 küme oluştur
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train)

# Her küme için en sık görülen gerçek rakamı bul
cluster_labels = {}
for i in range(10):
    mask = (kmeans.labels_ == i)
    cluster_labels[i] = np.bincount(y_train[mask]).argmax()

# Test seti tahminleri
y_pred_kmeans = [cluster_labels[label] for label in kmeans.predict(X_test)]

# Doğruluk
from sklearn.metrics import accuracy_score
acc_kmeans = accuracy_score(y_test, y_pred_kmeans)
print(f"Kümeleme Doğruluk: {acc_kmeans:.2f}")

pca_2d = PCA(n_components=2)
X_test_2d = pca_2d.fit_transform(X_test)

plt.figure(figsize=(8,6))
plt.scatter(
    X_test_2d[:,0], 
    X_test_2d[:,1], 
    c=kmeans.predict(X_test),  # KMeans küme etiketleri
    cmap='tab10',               # 10 farklı renk
    s=10                        # nokta boyutu
)
plt.title("MNIST Kümeleme (KMeans, PCA 2D)", fontsize=14)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()