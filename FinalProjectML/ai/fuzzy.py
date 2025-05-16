# Import các thư viện cần thiết
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa dữ liệu để có mean=0 và std=1
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    silhouette_score, davies_bouldin_score, adjusted_rand_score
)  # Các chỉ số để đánh giá chất lượng phân cụm
from scipy.stats import mode  # Dùng để xác định nhãn phổ biến trong cụm (nếu cần)
import pandas as pd  # Xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt  # Vẽ biểu đồ trực quan
import seaborn as sns  # Vẽ heatmap, biểu đồ đẹp hơn
import numpy as np  # Thư viện toán học nền tảng
import skfuzzy as fuzz  # Thư viện thực hiện Fuzzy C-Means

# Đọc dữ liệu từ file CSV (chú ý đường dẫn file có thể cần thay đổi nếu lỗi)
df_test = pd.read_csv(".\\data\\heart.csv")

# Tách dữ liệu thành đặc trưng (X) và nhãn thực tế (y)
X_CSV = df_test.drop(columns=["target"])      # X chứa các đặc trưng đầu vào
y_CSV_test = df_test["target"].values         # y chứa nhãn thực tế: 0 (không bệnh), 1 (có bệnh)

# Chuẩn hóa dữ liệu đầu vào giúp tăng độ hiệu quả khi phân cụm (tránh bị ảnh hưởng bởi scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_CSV)        # X_scaled là dữ liệu sau khi chuẩn hóa

# In dữ liệu chuẩn hóa (nếu cần xem, có thể xóa dòng này khi chạy chính thức)
print(X_scaled)

# ---------------------------------------------------------------
# Áp dụng thuật toán Fuzzy C-Means:
# - X_scaled.T: transpose vì cmeans yêu cầu shape là (features, samples)
# - 2: số lượng cụm
# - 2: hệ số m (m=2 là phổ biến, thể hiện mức độ "mềm" của phân cụm)
# - error=0.005: ngưỡng dừng thuật toán
# - maxiter=1000: số lần lặp tối đa
c, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_scaled.T, 2, 2, error=0.005, maxiter=1000)

# Với mỗi điểm dữ liệu, ta lấy nhãn của cụm có xác suất (membership) cao nhất
y_fuzzy = np.argmax(u, axis=0)

# Vì nhãn cụm là unsupervised nên có thể bị đảo 0 ↔ 1 → kiểm tra để chọn nhãn tối ưu
acc_original = accuracy_score(y_CSV_test, y_fuzzy)
acc_flipped = accuracy_score(y_CSV_test, 1 - y_fuzzy)

# Đảo nhãn nếu đảo cho độ chính xác cao hơn
if acc_flipped > acc_original:
    y_fuzzy = 1 - y_fuzzy

# ---------------------------------------------------------------
# In ma trận nhầm lẫn giữa nhãn thực và nhãn phân cụm
print("Fuzzy C-Means Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_fuzzy))

# In báo cáo phân loại: precision, recall, f1-score
print("\nFuzzy C-Means Classification Report:")
print(classification_report(y_CSV_test, y_fuzzy))

# ---------------------------------------------------------------
# Vẽ heatmap trực quan hóa ma trận nhầm lẫn
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_CSV_test, y_fuzzy), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Fuzzy C-Means Clustering")  # Tiêu đề
plt.xlabel("Predicted Labels")  # Nhãn trục X
plt.ylabel("True Labels")       # Nhãn trục Y
plt.show()

# ---------------------------------------------------------------
# Đánh giá chất lượng phân cụm bằng các chỉ số khách quan

# Silhouette Score: càng gần 1 thì cụm càng rõ ràng
sil_score = silhouette_score(X_scaled, y_fuzzy)

# Davies-Bouldin Index: càng thấp càng tốt, cho thấy khoảng cách cụm rõ ràng
dbi_score = davies_bouldin_score(X_scaled, y_fuzzy)

# In kết quả các chỉ số đánh giá
print(f"Silhouette Score: {sil_score}")
print(f"Davies-Bouldin Index: {dbi_score}")
