# Import các thư viện cần thiết
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa dữ liệu để mỗi đặc trưng có mean=0 và std=1
from sklearn.cluster import KMeans  # Thuật toán phân cụm KMeans
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    silhouette_score, davies_bouldin_score, adjusted_rand_score
)  # Các chỉ số để đánh giá hiệu quả phân cụm
from scipy.stats import mode  # Dùng để tìm nhãn phổ biến nhất trong cụm (nếu cần xử lý mapping cụm)
import pandas as pd  # Xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt  # Vẽ biểu đồ
import seaborn as sns  # Thư viện trực quan hóa nâng cao

# Đọc dữ liệu từ file CSV (thay đổi đường dẫn nếu cần)
df_test = pd.read_csv(".\data\heart.csv")  # Thay đường dẫn nếu lỗi

# Tách dữ liệu thành phần đặc trưng (X) và nhãn mục tiêu (y)
X_CSV = df_test.drop(columns=["target"])  # X là tất cả các cột trừ 'target'
y_CSV_test = df_test["target"].values     # y là nhãn thực: 0 (không bệnh), 1 (có bệnh)

# Chuẩn hóa dữ liệu về cùng thang đo để tăng độ chính xác khi tính khoảng cách trong KMeans
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_CSV)  # Trả về dữ liệu đã chuẩn hóa (z-score)

# In ra dữ liệu đã chuẩn hóa (có thể tắt dòng này nếu không cần xem chi tiết)
print(X_scaled)

# Khởi tạo mô hình KMeans với 2 cụm (giả định phân chia thành 2 nhóm: bệnh và không bệnh)
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)  # Dự đoán cụm cho từng mẫu

# Lưu ý: Vì KMeans là unsupervised, nhãn cụm (0/1) có thể không khớp với nhãn thực tế.
# Do đó, ta tính độ chính xác cho cả trường hợp giữ nguyên và đảo nhãn.
acc_original = accuracy_score(y_CSV_test, y_kmeans)
acc_flipped = accuracy_score(y_CSV_test, 1 - y_kmeans)

# Nếu đảo nhãn cho độ chính xác cao hơn thì thực hiện đảo nhãn
if acc_flipped > acc_original:
    y_kmeans = 1 - y_kmeans

# In ma trận nhầm lẫn để thấy rõ số mẫu đúng/sai giữa nhãn thật và nhãn dự đoán
print("KMeans Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_kmeans))

# In báo cáo phân loại bao gồm precision, recall, f1-score cho mỗi nhãn
print("\nKMeans Classification Report:")
print(classification_report(y_CSV_test, y_kmeans))

# Tính các chỉ số đánh giá chất lượng phân cụm
sil_score = silhouette_score(X_scaled, y_kmeans)  # Silhouette Score: càng gần 1 càng tốt
dbi_score = davies_bouldin_score(X_scaled, y_kmeans)  # Davies-Bouldin Index: càng thấp càng tốt

# In các chỉ số
print(f"Silhouette Score: {sil_score}")
print(f"Davies-Bouldin Index: {dbi_score}")

# Trực quan hóa ma trận nhầm lẫn dưới dạng heatmap
plt.figure(figsize=(6, 4))  # Kích thước biểu đồ
sns.heatmap(confusion_matrix(y_CSV_test, y_kmeans), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for KMeans Clustering")  # Tiêu đề biểu đồ
plt.xlabel("Predicted Labels")  # Nhãn trục x
plt.ylabel("True Labels")       # Nhãn trục y
plt.show()
