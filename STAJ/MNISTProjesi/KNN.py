from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 28x28 -> 784 boyut ve normalize et
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"kNN Doğruluk: {acc_knn:.2f}")

#yanlış sınıflanan örnekler
wrong_idx = np.where(y_pred_knn != y_test)[0][:25]  # İlk 25 yanlış
plt.figure(figsize=(8,8))
for i, idx in enumerate(wrong_idx):
    plt.subplot(5,5,i+1)
    plt.imshow(X_test[idx].reshape(28,28), cmap="gray")
    plt.title(f"T:{y_test[idx]} P:{y_pred_knn[idx]}")
    plt.axis("off")
plt.suptitle("Yanlış Sınıflanan Örnekler (kNN)", fontsize=16)
plt.show()