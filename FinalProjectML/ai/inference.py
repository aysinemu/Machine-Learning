# Import các thư viện cần thiết
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa dữ liệu
from pyclustering.cluster.kmedoids import kmedoids  # Thuật toán K-Medoids
from sklearn.mixture import GaussianMixture  # Gaussian Mixture Model (GMM)
from sklearn.cluster import AgglomerativeClustering, KMeans  # Clustering: Agglomerative, KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, silhouette_score, davies_bouldin_score  # Đánh giá phân cụm
import skfuzzy as fuzz  # Fuzzy C-Means clustering
import pandas as pd  # Xử lý dữ liệu dạng bảng
import numpy as np  # Xử lý số học, mảng
import matplotlib.pyplot as plt  # Vẽ biểu đồ
import seaborn as sns  # Vẽ biểu đồ nâng cao
from scipy.stats import mode
def predict_from_features(Test):
    # Đọc dữ liệu từ file CSV
    df_test = pd.read_csv(r"M:\ML\FinalProjectML\ai\data\heart.csv")# Đường dẫn đến file dữ liệu (có thể thay đổi theo máy bạn)

    # Tách biến đặc trưng (X) và nhãn mục tiêu (y)
    X_CSV = df_test.drop(columns=["target"])  # Bỏ cột "target" để lấy dữ liệu đầu vào
    y_CSV_test = df_test["target"].values  # Lấy cột "target" làm nhãn

    # Chuẩn hóa dữ liệu (mean=0, std=1) để các thuật toán clustering hoạt động tốt hơn
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_CSV)
    print(X_scaled)  # In dữ liệu sau chuẩn hóa

    # --------- KMeans Clustering ---------
    kmeans = KMeans(n_clusters=2, random_state=42)  # Khởi tạo KMeans với 2 cụm
    y_kmeans = kmeans.fit_predict(X_scaled)  # Huấn luyện và dự đoán nhãn cụm

    # Kiểm tra xem nhãn có bị đảo không (vì clustering không biết nhãn đúng)
    acc_original = accuracy_score(y_CSV_test, y_kmeans)
    acc_flipped = accuracy_score(y_CSV_test, 1 - y_kmeans)  # Đảo nhãn 0↔1

    # Nếu nhãn đảo cho kết quả chính xác hơn thì đảo lại nhãn phân cụm
    if acc_flipped > acc_original:
        y_kmeans = 1 - y_kmeans

    # In ma trận nhầm lẫn và báo cáo phân loại để đánh giá
    print("KMeans Confusion Matrix:")
    print(confusion_matrix(y_CSV_test, y_kmeans))
    print("\nKMeans Classification Report:")
    print(classification_report(y_CSV_test, y_kmeans))

    # Tính các chỉ số đánh giá phân cụm nội tại (không cần nhãn)
    sil_score = silhouette_score(X_scaled, y_kmeans)  # Silhouette Score (điểm càng cao càng tốt)
    dbi_score = davies_bouldin_score(X_scaled, y_kmeans)  # Davies-Bouldin Index (điểm càng thấp càng tốt)

    print(f"Silhouette Score: {sil_score}")
    print(f"Davies-Bouldin Index: {dbi_score}")

    # --------- K-Means++ (phiên bản khởi tạo tốt hơn) ---------
    kmeans_plus = KMeans(n_clusters=2, init='k-means++', random_state=42)
    y_kmeans_plus = kmeans_plus.fit_predict(X_scaled)

    # Tương tự xử lý đảo nhãn như trên
    acc_original = accuracy_score(y_CSV_test, y_kmeans_plus)
    acc_flipped = accuracy_score(y_CSV_test, 1 - y_kmeans_plus)
    if acc_flipped > acc_original:
        y_kmeans_plus = 1 - y_kmeans_plus

    # Đánh giá và in kết quả
    print("K-Means++ Confusion Matrix:")
    print(confusion_matrix(y_CSV_test, y_kmeans_plus))
    print("\nK-Means++ Classification Report:")
    print(classification_report(y_CSV_test, y_kmeans_plus))
    sil_score = silhouette_score(X_scaled, y_kmeans_plus)
    dbi_score = davies_bouldin_score(X_scaled, y_kmeans_plus)
    print(f"Silhouette Score: {sil_score}")
    print(f"Davies-Bouldin Index: {dbi_score}")

    # --------- Fuzzy C-Means ---------
    # fuzz.cluster.cmeans nhận dữ liệu có dạng mảng shape (features, samples) => transpose X_scaled
    c, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_scaled.T, 2, 2, error=0.005, maxiter=1000)

    # u là ma trận xác suất thuộc về từng cụm, lấy cụm có xác suất cao nhất
    y_fuzzy = np.argmax(u, axis=0)

    # Xử lý đảo nhãn như trên
    acc_original = accuracy_score(y_CSV_test, y_fuzzy)
    acc_flipped = accuracy_score(y_CSV_test, 1 - y_fuzzy)
    if acc_flipped > acc_original:
        y_fuzzy = 1 - y_fuzzy

    print("Fuzzy C-Means Confusion Matrix:")
    print(confusion_matrix(y_CSV_test, y_fuzzy))
    print("\nFuzzy C-Means Classification Report:")
    print(classification_report(y_CSV_test, y_fuzzy))
    sil_score = silhouette_score(X_scaled, y_fuzzy)
    dbi_score = davies_bouldin_score(X_scaled, y_fuzzy)
    print(f"Silhouette Score: {sil_score}")
    print(f"Davies-Bouldin Index: {dbi_score}")

    # --------- K-Medoids ---------
    initial_medoids = [0, 1]  # Chọn 2 điểm đầu tiên làm medoid ban đầu

    kmedoids_instance = kmedoids(X_scaled, initial_medoids)
    kmedoids_instance.process()  # Thực hiện phân cụm

    clusters = kmedoids_instance.get_clusters()  # Lấy danh sách các cụm

    # Khởi tạo mảng nhãn cho dữ liệu
    y_kmedoids = np.zeros(X_scaled.shape[0])

    # Gán nhãn cụm cho từng điểm dữ liệu
    for idx, cluster in enumerate(clusters):
        y_kmedoids[cluster] = idx

    # Xử lý đảo nhãn nếu cần
    acc_original = accuracy_score(y_CSV_test, y_kmedoids)
    acc_flipped = accuracy_score(y_CSV_test, 1 - y_kmedoids)
    if acc_flipped > acc_original:
        y_kmedoids = 1 - y_kmedoids

    print("K-Medoids Confusion Matrix:")
    print(confusion_matrix(y_CSV_test, y_kmedoids))
    print("\nK-Medoids Classification Report:")
    print(classification_report(y_CSV_test, y_kmedoids))
    sil_score = silhouette_score(X_scaled, y_kmedoids)
    dbi_score = davies_bouldin_score(X_scaled, y_kmedoids)
    print(f"Silhouette Score: {sil_score}")
    print(f"Davies-Bouldin Index: {dbi_score}")

    # --------- Gaussian Mixture Model (GMM) ---------
    gmm = GaussianMixture(n_components=2, random_state=42)
    y_gmm = gmm.fit_predict(X_scaled)

    # Xử lý đảo nhãn nếu cần
    acc_original = accuracy_score(y_CSV_test, y_gmm)
    acc_flipped = accuracy_score(y_CSV_test, 1 - y_gmm)
    if acc_flipped > acc_original:
        y_gmm = 1 - y_gmm

    print("GMM Confusion Matrix:")
    print(confusion_matrix(y_CSV_test, y_gmm))
    print("\nGMM Classification Report:")
    print(classification_report(y_CSV_test, y_gmm))
    sil_score = silhouette_score(X_scaled, y_gmm)
    dbi_score = davies_bouldin_score(X_scaled, y_gmm)
    print(f"Silhouette Score: {sil_score}")
    print(f"Davies-Bouldin Index: {dbi_score}")

    # --------- Agglomerative Hierarchical Clustering ---------
    agg_clustering = AgglomerativeClustering(n_clusters=2)
    y_agg = agg_clustering.fit_predict(X_scaled)

    # Xử lý đảo nhãn nếu cần
    acc_original = accuracy_score(y_CSV_test, y_agg)
    acc_flipped = accuracy_score(y_CSV_test, 1 - y_agg)
    if acc_flipped > acc_original:
        y_agg = 1 - y_agg

    print("Agglomerative Hierarchical Clustering Confusion Matrix:")
    print(confusion_matrix(y_CSV_test, y_agg))
    print("\nAgglomerative Hierarchical Clustering Classification Report:")
    print(classification_report(y_CSV_test, y_agg))
    sil_score = silhouette_score(X_scaled, y_agg)
    dbi_score = davies_bouldin_score(X_scaled, y_agg)
    print(f"Silhouette Score: {sil_score}")
    print(f"Davies-Bouldin Index: {dbi_score}")

    # --------- Phần dự đoán mẫu mới ---------
    # Danh sách các đặc trưng trong dataset để bạn dễ hiểu ý nghĩa từng cột
    features = [
        {'name': 'age', 'description': 'Age of the patient (in years)', 'type': 'Integer', 'units': 'years'},
        {'name': 'sex', 'description': 'Gender of the patient (1 = male, 0 = female)', 'type': 'Categorical', 'units': 'None'},
        {'name': 'cp', 'description': 'Chest pain type (4 types)', 'type': 'Categorical', 'units': 'None'},
        {'name': 'trestbps', 'description': 'Resting blood pressure (mm Hg)', 'type': 'Integer', 'units': 'mm Hg'},
        {'name': 'chol', 'description': 'Serum cholesterol level (mg/dl)', 'type': 'Integer', 'units': 'mg/dl'},
        {'name': 'fbs', 'description': 'Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)', 'type': 'Categorical', 'units': 'None'},
        {'name': 'restecg', 'description': 'Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: LVH)', 'type': 'Categorical', 'units': 'None'},
        {'name': 'thalach', 'description': 'Maximum heart rate achieved (bpm)', 'type': 'Integer', 'units': 'bpm'},
        {'name': 'exang', 'description': 'Exercise induced angina (1 = yes, 0 = no)', 'type': 'Categorical', 'units': 'None'},
        {'name': 'oldpeak', 'description': 'ST depression induced by exercise relative to rest', 'type': 'Integer', 'units': 'None'},
        {'name': 'slope', 'description': 'Slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)', 'type': 'Categorical', 'units': 'None'},
        {'name': 'ca', 'description': 'Number of major vessels (0-3) colored by fluoroscopy', 'type': 'Integer', 'units': 'None'},
        {'name': 'thal', 'description': 'Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)', 'type': 'Categorical', 'units': 'None'}
    ]

    # Chuẩn hóa mẫu mới theo scaler đã fit trên dữ liệu huấn luyện
    scaler.fit(X_CSV)  # Fit scaler lại trên dữ liệu gốc (nếu cần)
    X_new = scaler.transform(Test)  # Chuẩn hóa dữ liệu mẫu mới

    print("\nDữ liệu mẫu mới đã chuẩn hóa:")
    print(X_new)

    # Giả sử bạn có dữ liệu mới X_new để dự đoán cụm (cluster)
    kmeans_pred = kmeans.predict(X_new)               # Dự đoán cụm bằng KMeans
    kmeans_plus_pred = kmeans_plus.predict(X_new)     # Dự đoán cụm bằng KMeans++ (phiên bản cải tiến)
    kmedoids_pred = kmedoids_instance.predict(X_new)  # Dự đoán cụm bằng K-Medoids
    gmm_pred = gmm.predict(X_new)                      # Dự đoán cụm bằng Gaussian Mixture Model (GMM)

    # Chuyển đổi centroids về dạng (số cụm, số đặc trưng)
    # vì c ban đầu có shape (features, clusters), nên dùng c.T để chuyển thành (clusters, features)
    centroids = c.T  

    # Tính khoảng cách Euclid từ mỗi điểm dữ liệu trong Test đến từng centroid
    # Test có shape (1, 13) - 1 mẫu, 13 đặc trưng
    # centroids.T có shape (13, 2), dùng np.newaxis để broadcast và tính toán khoảng cách đúng
    distances = np.linalg.norm(Test[:, np.newaxis, :] - centroids.T[np.newaxis, :, :], axis=2)  # Kết quả shape (1, n_clusters)

    # Tính ma trận membership cho Fuzzy C-Means
    m = 2  # hệ số fuzziness (độ mờ)
    min_dist = np.min(distances, axis=1, keepdims=True) + 1e-6  # Tránh chia cho 0 bằng cách cộng epsilon nhỏ
    u_new = 1 / (1 + (distances / min_dist) ** (2 / (m - 1)))  # Công thức membership của Fuzzy C-Means

    # Dự đoán cụm bằng cách lấy cụm có giá trị membership lớn nhất
    predicted_clusters = np.argmax(u_new, axis=1)

    # Đảo ngược nhãn dự đoán
    # kmeans_pred = 1 - np.array(kmeans_pred)
    # kmeans_plus_pred = 1 - np.array(kmeans_plus_pred)
    # kmedoids_pred = 1 - np.array(kmedoids_pred)
    # gmm_pred = 1 - np.array(gmm_pred)
    # predicted_clusters = 1 - np.array(predicted_clusters)

    # In kết quả dự đoán sau khi đảo nhãn
    print("kmeans_pred:", kmeans_pred)
    print("kmeans_plus_pred:", kmeans_plus_pred)
    print("kmedoids_pred:", kmedoids_pred)
    print("gmm_pred:", gmm_pred)
    print("predicted_clusters:", predicted_clusters)
    print("Predictions done!")

    # Tạo DataFrame chứa các kết quả dự đoán của các thuật toán clustering
    labels_Test = pd.DataFrame({
        'kmeans': kmeans_pred,
        'kmeans_plus': kmeans_plus_pred,
        'kmedoids': kmedoids_pred,
        'Fuzzy': predicted_clusters,
        'gmm': gmm_pred
    })

    # Thực hiện hard voting (bỏ phiếu đa số) để kết luận dự đoán cuối cùng
    final = mode(labels_Test.values, axis=1)[0].flatten()  # mode trả về (mode, count), lấy mode

    # In kết quả cuối cùng: 0 là không có nguy cơ bệnh tim, 1 là có nguy cơ
    if final[0] == 0:
        print("Result: NOT at risk of heart disease")
        return False
    else:
        print("Result: AT RISK of heart disease")
        return True
