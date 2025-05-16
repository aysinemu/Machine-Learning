# Import các thư viện cần thiết
from sklearn.preprocessing import StandardScaler              # Chuẩn hóa dữ liệu
from pyclustering.cluster.kmedoids import kmedoids           # Thuật toán K-Medoids
from sklearn.mixture import GaussianMixture                  # Gaussian Mixture Model (GMM)
from sklearn.cluster import AgglomerativeClustering, KMeans  # KMeans và Agglomerative Clustering
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, silhouette_score,
                             davies_bouldin_score, adjusted_rand_score)  # Các thước đo đánh giá
from scipy.stats import mode                                  # Tìm mode (đa số phiếu trong voting)
import skfuzzy as fuzz                                        # Fuzzy C-Means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
df_test = pd.read_csv(".\\data\\heart.csv")  # Đổi đường dẫn nếu cần
X_CSV = df_test.drop(columns=["target"])     # Dữ liệu đầu vào (không gồm nhãn)
y_CSV_test = df_test["target"].values        # Nhãn thật (ground truth)

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_CSV)
print(X_scaled)

### ----- KMeans ----- ###
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Đảo nhãn nếu cần (0 ↔ 1)
acc_original = accuracy_score(y_CSV_test, y_kmeans)
acc_flipped = accuracy_score(y_CSV_test, 1 - y_kmeans)
if acc_flipped > acc_original:
    y_kmeans = 1 - y_kmeans

# Đánh giá kết quả
print("KMeans Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_kmeans))
print("\nKMeans Classification Report:")
print(classification_report(y_CSV_test, y_kmeans))
print("Silhouette Score:", silhouette_score(X_scaled, y_kmeans))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, y_kmeans))

### ----- KMeans++ (Khởi tạo thông minh hơn) ----- ###
kmeans_plus = KMeans(n_clusters=2, init='k-means++', random_state=42)
y_kmeans_plus = kmeans_plus.fit_predict(X_scaled)
acc_original = accuracy_score(y_CSV_test, y_kmeans_plus)
acc_flipped = accuracy_score(y_CSV_test, 1 - y_kmeans_plus)
if acc_flipped > acc_original:
    y_kmeans_plus = 1 - y_kmeans_plus

print("K-Means++ Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_kmeans_plus))
print("\nK-Means++ Classification Report:")
print(classification_report(y_CSV_test, y_kmeans_plus))
print("Silhouette Score:", silhouette_score(X_scaled, y_kmeans_plus))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, y_kmeans_plus))

### ----- Fuzzy C-Means ----- ###
c, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_scaled.T, 2, 2, error=0.005, maxiter=1000)
y_fuzzy = np.argmax(u, axis=0)
acc_original = accuracy_score(y_CSV_test, y_fuzzy)
acc_flipped = accuracy_score(y_CSV_test, 1 - y_fuzzy)
if acc_flipped > acc_original:
    y_fuzzy = 1 - y_fuzzy

print("Fuzzy C-Means Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_fuzzy))
print("\nFuzzy C-Means Classification Report:")
print(classification_report(y_CSV_test, y_fuzzy))
print("Silhouette Score:", silhouette_score(X_scaled, y_fuzzy))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, y_fuzzy))

### ----- K-Medoids ----- ###
initial_medoids = [0, 1]  # Khởi tạo hai điểm medoid ban đầu
kmedoids_instance = kmedoids(X_scaled, initial_medoids)
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()

# Gán nhãn cho mỗi điểm dữ liệu
y_kmedoids = np.zeros(X_scaled.shape[0])
for idx, cluster in enumerate(clusters):
    y_kmedoids[cluster] = idx

acc_original = accuracy_score(y_CSV_test, y_kmedoids)
acc_flipped = accuracy_score(y_CSV_test, 1 - y_kmedoids)
if acc_flipped > acc_original:
    y_kmedoids = 1 - y_kmedoids

print("K-Medoids Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_kmedoids))
print("\nK-Medoids Classification Report:")
print(classification_report(y_CSV_test, y_kmedoids))
print("Silhouette Score:", silhouette_score(X_scaled, y_kmedoids))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, y_kmedoids))

### ----- Gaussian Mixture Model (GMM) ----- ###
gmm = GaussianMixture(n_components=2, random_state=42)
y_gmm = gmm.fit_predict(X_scaled)
acc_original = accuracy_score(y_CSV_test, y_gmm)
acc_flipped = accuracy_score(y_CSV_test, 1 - y_gmm)
if acc_flipped > acc_original:
    y_gmm = 1 - y_gmm

print("GMM Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_gmm))
print("\nGMM Classification Report:")
print(classification_report(y_CSV_test, y_gmm))
print("Silhouette Score:", silhouette_score(X_scaled, y_gmm))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, y_gmm))

### ----- Agglomerative Clustering ----- ###
agg_clustering = AgglomerativeClustering(n_clusters=2)
y_agg = agg_clustering.fit_predict(X_scaled)
acc_original = accuracy_score(y_CSV_test, y_agg)
acc_flipped = accuracy_score(y_CSV_test, 1 - y_agg)
if acc_flipped > acc_original:
    y_agg = 1 - y_agg

print("Agglomerative Clustering Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_agg))
print("\nAgglomerative Clustering Classification Report:")
print(classification_report(y_CSV_test, y_agg))
print("Silhouette Score:", silhouette_score(X_scaled, y_agg))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, y_agg))

### ----- Hard Voting từ các phương pháp phân cụm ----- ###
labels = pd.DataFrame({
    'kmeans': y_kmeans,
    'kmeans_plus': y_kmeans_plus,
    'kmedoids': y_kmedoids,
    'Fuzzy': y_fuzzy,
    'agg': y_agg,
    'gmm': y_gmm
})

# Hàm đảo nhãn thông minh dựa trên accuracy
def best_label_match(pred, y_true_test):
    acc1 = accuracy_score(y_true_test, pred)
    acc2 = accuracy_score(y_true_test, 1 - pred)
    if acc2 - acc1 > 0.05:  # Ngưỡng có thể điều chỉnh
        return 1 - pred
    else:
        return pred

# Áp dụng đảo nhãn cho từng cột trong DataFrame
for col in labels.columns:
    labels[col] = best_label_match(labels[col], y_CSV_test)

# Hard voting (chọn nhãn phổ biến nhất trong 6 mô hình)
final_pred = mode(labels.values, axis=1)[0].flatten()

print("Final Confusion Matrix:")
print(confusion_matrix(y_CSV_test, final_pred))
print("\nFinal Classification Report:")
print(classification_report(y_CSV_test, final_pred))
print("Silhouette Score:", silhouette_score(X_scaled, final_pred))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, final_pred))

# Vẽ biểu đồ heatmap cho confusion matrix cuối cùng
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_CSV_test, final_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Final Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
