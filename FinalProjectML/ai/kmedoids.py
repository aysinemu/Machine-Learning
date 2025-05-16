# ---------------------------------------------------------------
# Import các thư viện cần thiết
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa dữ liệu để dữ liệu có mean=0, std=1
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    silhouette_score, davies_bouldin_score, adjusted_rand_score
)  # Các chỉ số đánh giá chất lượng phân cụm
from scipy.stats import mode  # Tìm nhãn phổ biến trong mỗi cụm nếu cần
import pandas as pd  # Đọc và xử lý dữ liệu dạng bảng (DataFrame)
import matplotlib.pyplot as plt  # Vẽ biểu đồ
import seaborn as sns  # Vẽ biểu đồ nâng cao
from pyclustering.cluster.kmedoids import kmedoids
import numpy as np

# Lưu ý: Bạn cần import hoặc định nghĩa hàm kmedoids hoặc thư viện hỗ trợ K-Medoids,
# ví dụ: from pyclustering.cluster.kmedoids import kmedoids
# Hoặc bạn đã định nghĩa hàm kmedoids riêng

# ---------------------------------------------------------------
# Bước 1: Đọc dữ liệu từ file CSV
df_test = pd.read_csv(".\data\heart.csv")  # Đọc dữ liệu, sửa đường dẫn nếu lỗi

# ---------------------------------------------------------------
# Bước 2: Tách dữ liệu thành đặc trưng (X) và nhãn thực (y)
X_CSV = df_test.drop(columns=["target"])     # Tất cả cột trừ 'target' là đặc trưng đầu vào
y_CSV_test = df_test["target"].values        # Nhãn thật: 0 (không bệnh), 1 (có bệnh)

# ---------------------------------------------------------------
# Bước 3: Chuẩn hóa dữ liệu để các đặc trưng cùng thang đo, tránh bias
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_CSV)       # Dữ liệu chuẩn hóa

# In ra dữ liệu chuẩn hóa để kiểm tra (có thể bỏ khi chạy chính thức)
print(X_scaled)

# ---------------------------------------------------------------
# Bước 4: Chọn điểm khởi tạo cho K-Medoids
# K-Medoids cần chọn trước 1 số điểm làm "medoid" - đại diện cụm
# Ở đây giả sử chọn 2 điểm đầu tiên làm medoid ban đầu (có thể chọn khác)
initial_medoids = [0, 1]

# ---------------------------------------------------------------
# Bước 5: Khởi tạo và chạy thuật toán K-Medoids
# kmedoids là đối tượng của thuật toán, truyền vào dữ liệu và medoids khởi tạo
kmedoids_instance = kmedoids(X_scaled, initial_medoids)
kmedoids_instance.process()  # Thực thi thuật toán

# ---------------------------------------------------------------
# Bước 6: Lấy kết quả phân cụm
clusters = kmedoids_instance.get_clusters()  # Trả về list các cụm, mỗi cụm là list index các điểm dữ liệu

# Khởi tạo mảng nhãn phân cụm, ban đầu để 0 cho tất cả điểm
y_kmedoids = np.zeros(X_scaled.shape[0])

# Gán nhãn cụm cho từng điểm dữ liệu dựa trên kết quả phân cụm
for idx, cluster in enumerate(clusters):
    for data_index in cluster:
        y_kmedoids[data_index] = idx

# ---------------------------------------------------------------
# Bước 7: Kiểm tra nhãn phân cụm có trùng khớp nhãn thật hay không
# Vì nhãn cụm không cố định, có thể bị đảo ngược 0↔1 nên ta so sánh 2 trường hợp

acc_original = accuracy_score(y_CSV_test, y_kmedoids)       # Độ chính xác nhãn gốc
acc_flipped = accuracy_score(y_CSV_test, 1 - y_kmedoids)     # Độ chính xác nếu đảo nhãn

# Nếu đảo nhãn cho độ chính xác cao hơn, thực hiện đảo nhãn
if acc_flipped > acc_original:
    y_kmedoids = 1 - y_kmedoids

# ---------------------------------------------------------------
# Bước 8: In kết quả phân cụm ra màn hình
print("K-Medoids Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_kmedoids))  # Ma trận nhầm lẫn

print("\nK-Medoids Classification Report:")
print(classification_report(y_CSV_test, y_kmedoids))  # Báo cáo các chỉ số precision, recall, f1-score

# ---------------------------------------------------------------
# Bước 9: Trực quan hóa ma trận nhầm lẫn bằng biểu đồ heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_CSV_test, y_kmedoids), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for K-Medoids Clustering")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# ---------------------------------------------------------------
# Bước 10: Đánh giá chất lượng phân cụm với các chỉ số không giám sát

# Silhouette Score: đo độ tách biệt và kết dính của các cụm (giá trị gần 1 càng tốt)
sil_score = silhouette_score(X_scaled, y_kmedoids)

# Davies-Bouldin Index: chỉ số đánh giá cụm, càng nhỏ càng tốt
dbi_score = davies_bouldin_score(X_scaled, y_kmedoids)

# In ra các chỉ số đánh giá
print(f"Silhouette Score: {sil_score}")
print(f"Davies-Bouldin Index: {dbi_score}")
