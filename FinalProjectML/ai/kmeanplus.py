# ---------------------------------------------------------------
# Import các thư viện cần thiết
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa dữ liệu để có mean=0, std=1
from sklearn.cluster import KMeans  # Sử dụng thuật toán K-Means để phân cụm
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    silhouette_score, davies_bouldin_score, adjusted_rand_score
)  # Các chỉ số đánh giá chất lượng phân cụm
from scipy.stats import mode  # Được dùng để xác định nhãn phổ biến trong cụm nếu cần
import pandas as pd  # Đọc và xử lý dữ liệu dưới dạng bảng (DataFrame)
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ
import seaborn as sns  # Thư viện vẽ biểu đồ nâng cao (dựa trên matplotlib)

# ---------------------------------------------------------------
# Bước 1: Đọc dữ liệu
# Dữ liệu bệnh tim với đặc trưng đầu vào và cột 'target' là nhãn thực
df_test = pd.read_csv(".\data\heart.csv")  # Thay đường dẫn nếu lỗi

# ---------------------------------------------------------------
# Bước 2: Tách dữ liệu thành X (feature) và y (label)
X_CSV = df_test.drop(columns=["target"])     # X là ma trận đặc trưng (features)
y_CSV_test = df_test["target"].values        # y là nhãn thật: 0 = không bệnh, 1 = có bệnh

# ---------------------------------------------------------------
# Bước 3: Chuẩn hóa dữ liệu để tránh bias do đơn vị đo lường khác nhau
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_CSV)       # Sau khi chuẩn hóa, mỗi cột có mean=0 và std=1

# In dữ liệu đã chuẩn hóa (có thể xóa dòng này khi chạy chính thức)
print(X_scaled)

# ---------------------------------------------------------------
# Bước 4: Áp dụng K-Means++ (cải tiến khởi tạo so với K-Means thông thường)
# - n_clusters=2: muốn chia dữ liệu thành 2 cụm (tương ứng với 0/1)
# - init='k-means++': cải thiện khởi tạo centroids để tránh kết quả kém
# - random_state=42: để kết quả có thể lặp lại (tái tạo)
kmeans_plus = KMeans(n_clusters=2, init='k-means++', random_state=42)
y_kmeans_plus = kmeans_plus.fit_predict(X_scaled)  # Nhãn phân cụm trả về dưới dạng mảng 0 hoặc 1

# ---------------------------------------------------------------
# Bước 5: Vì nhãn cụm không có thứ tự cố định (có thể bị đảo ngược)
# So sánh nhãn phân cụm với nhãn thật để xác định có cần đảo hay không
acc_original = accuracy_score(y_CSV_test, y_kmeans_plus)       # Độ chính xác gốc
acc_flipped = accuracy_score(y_CSV_test, 1 - y_kmeans_plus)     # Độ chính xác nếu đảo nhãn

# Nếu đảo nhãn mà cho độ chính xác cao hơn thì thực hiện đảo
if acc_flipped > acc_original:
    y_kmeans_plus = 1 - y_kmeans_plus

# ---------------------------------------------------------------
# Bước 6: In kết quả đánh giá phân cụm
print("K-Means++ Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_kmeans_plus))  # Ma trận nhầm lẫn

print("\nK-Means++ Classification Report:")
print(classification_report(y_CSV_test, y_kmeans_plus))  # Báo cáo phân loại: precision, recall, f1

# ---------------------------------------------------------------
# Bước 7: Trực quan hóa ma trận nhầm lẫn bằng biểu đồ heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_CSV_test, y_kmeans_plus), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for K-Means++ Clustering")  # Tiêu đề
plt.xlabel("Predicted Labels")  # Nhãn trục X
plt.ylabel("True Labels")       # Nhãn trục Y
plt.show()

# ---------------------------------------------------------------
# Bước 8: Đánh giá chất lượng phân cụm bằng các chỉ số không giám sát

# Silhouette Score: đo mức độ dữ liệu được phân nhóm tốt như thế nào (gần 1 là tốt)
sil_score = silhouette_score(X_scaled, y_kmeans_plus)

# Davies-Bouldin Index: càng thấp càng tốt, thể hiện các cụm rõ ràng và cách biệt
dbi_score = davies_bouldin_score(X_scaled, y_kmeans_plus)

# In các chỉ số đánh giá
print(f"Silhouette Score: {sil_score}")
print(f"Davies-Bouldin Index: {dbi_score}")
