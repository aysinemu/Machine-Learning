# Import các thư viện cần thiết
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa dữ liệu để có cùng thang đo
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    silhouette_score, davies_bouldin_score, adjusted_rand_score
)  # Các chỉ số đánh giá phân cụm và độ chính xác
from scipy.stats import mode  # Tìm giá trị phổ biến (thường dùng để gán nhãn cụm)
import pandas as pd  # Xử lý dữ liệu dạng bảng (DataFrame)
import matplotlib.pyplot as plt  # Vẽ biểu đồ
import seaborn as sns  # Thư viện trực quan hóa nâng cao (heatmap,...)

# Đọc dữ liệu từ file CSV
df_test = pd.read_csv(".\data\heart.csv")  # Thay đường dẫn nếu lỗi

# Tách đặc trưng (X) và nhãn (y)
X_CSV = df_test.drop(columns=["target"])  # Đặc trưng đầu vào (loại bỏ cột 'target')
y_CSV_test = df_test["target"].values  # Nhãn thật: 0 (không bệnh), 1 (có bệnh)

# Chuẩn hóa dữ liệu để đảm bảo các đặc trưng có cùng đơn vị, tránh bias khi tính khoảng cách
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_CSV)  # Fit và transform dữ liệu đầu vào
print(X_scaled)  # In ra dữ liệu đã chuẩn hóa

# Phân cụm sử dụng Agglomerative Hierarchical Clustering (PHÂN CỤM KHÔNG GIÁM SÁT)
from sklearn.cluster import AgglomerativeClustering  # (Import bị thiếu ở trên)
agg_clustering = AgglomerativeClustering(n_clusters=2)  # Tạo mô hình với 2 cụm
y_agg = agg_clustering.fit_predict(X_scaled)  # Dự đoán nhãn phân cụm (0 hoặc 1)

# Đánh giá xem nhãn nào đúng, vì phân cụm không biết nhãn thật nên có thể bị đảo
acc_original = accuracy_score(y_CSV_test, y_agg)  # So sánh nhãn thật với nhãn dự đoán
acc_flipped = accuracy_score(y_CSV_test, 1 - y_agg)  # Đảo ngược nhãn cụm và so sánh lại

# Nếu nhãn đảo cho kết quả tốt hơn thì dùng nhãn đảo
if acc_flipped > acc_original:
    y_agg = 1 - y_agg

# Đánh giá kết quả phân cụm bằng các chỉ số phân loại
print("Agglomerative Hierarchical Clustering Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_agg))  # Ma trận nhầm lẫn
print("\nAgglomerative Hierarchical Clustering Classification Report:")
print(classification_report(y_CSV_test, y_agg))  # Precision, Recall, F1...

# Vẽ heatmap biểu diễn confusion matrix cho trực quan
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_CSV_test, y_agg), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Agglomerative Hierarchical Clustering")
plt.xlabel("Predicted Labels")  # Trục X: nhãn dự đoán
plt.ylabel("True Labels")  # Trục Y: nhãn thực tế
plt.show()

# Tính các chỉ số đánh giá cụm: càng cao càng tốt (Silhouette), càng thấp càng tốt (DBI)
sil_score = silhouette_score(X_scaled, y_agg)  # Silhouette Score: Đo độ tách biệt giữa các cụm
dbi_score = davies_bouldin_score(X_scaled, y_agg)  # Davies-Bouldin Index: Đo độ chồng lấn giữa các cụm

# In kết quả đánh giá
print(f"Silhouette Score: {sil_score}")
print(f"Davies-Bouldin Index: {dbi_score}")
