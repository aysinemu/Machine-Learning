# Import thư viện cần thiết
from sklearn.preprocessing import StandardScaler  # Dùng để chuẩn hóa dữ liệu
import pandas as pd  # Thư viện để xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ
from sklearn.decomposition import PCA  # Giảm chiều bằng Principal Component Analysis
from sklearn.manifold import TSNE  # Giảm chiều bằng t-SNE (phi tuyến)

# Đọc dữ liệu từ file CSV (dữ liệu bệnh tim)
df_test = pd.read_csv(".\data\heart.csv")  # Thay đường dẫn nếu lỗi

# Tách dữ liệu đầu vào (X) và nhãn đầu ra (y)
X_CSV = df_test.drop(columns=["target"])  # X là các đặc trưng (features), bỏ cột target
y_CSV_test = df_test["target"].values  # y là nhãn (0: không bệnh, 1: có bệnh)

# Chuẩn hóa dữ liệu đầu vào bằng StandardScaler (trung bình = 0, độ lệch chuẩn = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_CSV)  # Dữ liệu đầu vào đã được chuẩn hóa
print(X_scaled)  # In dữ liệu đã chuẩn hóa (chỉ để kiểm tra)

# Giảm chiều bằng PCA (Principal Component Analysis) về 2 chiều
pca = PCA(n_components=2)  # Giảm chiều còn 2 thành phần chính
X_pca = pca.fit_transform(X_scaled)  # Dữ liệu sau khi giảm chiều bằng PCA

# Giảm chiều bằng t-SNE (phù hợp trực quan hóa cụm dữ liệu phi tuyến)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)  # random_state để đảm bảo kết quả giống nhau mỗi lần chạy
X_tsne = tsne.fit_transform(X_scaled)  # Dữ liệu sau khi giảm chiều bằng t-SNE

# Tạo khung vẽ gồm 2 biểu đồ (1 hàng, 2 cột)
fig, axes = plt.subplots(1, 2, figsize=(22, 8))

# Vẽ biểu đồ phân tán với PCA
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_CSV_test, cmap='coolwarm', alpha=0.7)
axes[0].set_title('Dữ liệu giảm chiều bằng PCA')  # Tiêu đề biểu đồ
axes[0].set_xlabel('Thành phần chính 1')  # Trục X
axes[0].set_ylabel('Thành phần chính 2')  # Trục Y
axes[0].grid(True)  # Hiển thị lưới

# Vẽ biểu đồ phân tán với t-SNE
sc = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_CSV_test, cmap='coolwarm', alpha=0.7)
axes[1].set_title('Dữ liệu giảm chiều bằng t-SNE')  # Tiêu đề biểu đồ
axes[1].set_xlabel('Thành phần chính 1')  # Trục X
axes[1].set_ylabel('Thành phần chính 2')  # Trục Y
axes[1].grid(True)  # Hiển thị lưới

# Thêm thanh màu thể hiện nhãn (0: không bệnh, 1: bệnh)
cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=1)
cbar.set_label('Nhãn (0: không bệnh, 1: bệnh)')  # Nhãn của thanh màu

# Hiển thị toàn bộ biểu đồ
plt.show()