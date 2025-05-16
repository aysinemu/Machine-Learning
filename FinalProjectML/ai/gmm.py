# Import các thư viện cần thiết
from sklearn.preprocessing import StandardScaler  # Dùng để chuẩn hóa dữ liệu (mean = 0, std = 1)
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    silhouette_score, davies_bouldin_score, adjusted_rand_score  # Các thước đo đánh giá kết quả phân cụm
)
import pandas as pd  # Thư viện xử lý dữ liệu dạng bảng (DataFrame)
import matplotlib.pyplot as plt  # Vẽ biểu đồ
import seaborn as sns  # Vẽ biểu đồ đẹp hơn
from sklearn.mixture import GaussianMixture  # GMM – mô hình phân cụm dựa trên xác suất

# Đọc dữ liệu từ file CSV
df_test = pd.read_csv(".\data\heart.csv")  # Thay đường dẫn nếu lỗi

# Tách đặc trưng (X) và nhãn (y)
X_CSV = df_test.drop(columns=["target"])  # Bỏ cột 'target', giữ lại các đặc trưng để phân cụm
y_CSV_test = df_test["target"].values  # Lấy nhãn thực tế để đánh giá sau phân cụm (0 = không bệnh, 1 = có bệnh)

# Chuẩn hóa dữ liệu để đảm bảo các đặc trưng có cùng đơn vị (rất quan trọng cho thuật toán như GMM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_CSV)  # Fit và transform dữ liệu
print(X_scaled)  # In ra dữ liệu đã chuẩn hóa (có thể bỏ dòng này nếu không cần xem)

# Áp dụng GMM (Gaussian Mixture Model) với số cụm = 2 (vì có 2 lớp: bệnh và không bệnh)
gmm = GaussianMixture(n_components=2, random_state=42)  # Khởi tạo mô hình GMM
y_gmm = gmm.fit_predict(X_scaled)  # Huấn luyện và dự đoán cụm

# GMM có thể hoán đổi nhãn (cụm 0 có thể ứng với nhãn 1 và ngược lại),
# nên ta thử cả 2 cách gán nhãn và chọn cách nào có độ chính xác cao hơn
acc_original = accuracy_score(y_CSV_test, y_gmm)
acc_flipped = accuracy_score(y_CSV_test, 1 - y_gmm)

# Đảo nhãn nếu cần để khớp với nhãn thực tế
if acc_flipped > acc_original:
    y_gmm = 1 - y_gmm

# Đánh giá hiệu suất phân cụm bằng confusion matrix và classification report
print("GMM Confusion Matrix:")
print(confusion_matrix(y_CSV_test, y_gmm))  # Ma trận nhầm lẫn

print("\nGMM Classification Report:")
print(classification_report(y_CSV_test, y_gmm))  # Precision, Recall, F1-score cho từng lớp

# Vẽ heatmap thể hiện ma trận nhầm lẫn trực quan
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_CSV_test, y_gmm), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for GMM Clustering")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Tính các chỉ số đánh giá phân cụm không giám sát (không phụ thuộc vào nhãn thật)
sil_score = silhouette_score(X_scaled, y_gmm)  # Đánh giá độ "cô lập" giữa các cụm (cao hơn tốt hơn)
dbi_score = davies_bouldin_score(X_scaled, y_gmm)  # Đánh giá độ "giao thoa" giữa các cụm (thấp hơn tốt hơn)

# In kết quả các chỉ số
print(f"Silhouette Score: {sil_score}")
print(f"Davies-Bouldin Index: {dbi_score}")
