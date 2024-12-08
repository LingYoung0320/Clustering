import os
import time
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
seed_value = 2030
np.random.seed(seed_value)

class KMeans:
    def __init__(self, k=3, max_iter=100, tol=0.0001):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False), :]
        for i in range(self.max_iter):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = distances.argmin(axis=0)
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])
            delta = np.abs(self.centroids - new_centroids).max()
            if delta < self.tol:
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = distances.argmin(axis=0)
        return labels

# 创建结果存储文件夹和文件
os.makedirs("kmeans_pic", exist_ok=True)
result_file = "kmeans_2to20_result.txt"

# 确保之前的文件不存在
if os.path.exists(result_file):
    os.remove(result_file)

# 导入数据
data = np.loadtxt("normalized_data.txt", delimiter=" ")
X = data[:, :2]

# 聚类并输出结果
with open(result_file, "w") as f:
    for k in range(2, 21):
        start_time = time.time()
        model = KMeans(k=k)
        model.fit(X)
        labels = model.predict(X)
        end_time = time.time()

        # 计算指标
        SSE = sum(((X - model.centroids[labels]) ** 2).sum(axis=1))
        silhouette_scores = []
        for i in range(len(X)):
            same_cluster = X[labels == labels[i]]
            other_clusters = X[labels != labels[i]]
            a = np.mean(np.sqrt(((same_cluster - X[i]) ** 2).sum(axis=1))) if len(same_cluster) > 1 else 0
            b = np.min([np.mean(np.sqrt(((X[labels == j] - X[i]) ** 2).sum(axis=1))) for j in range(k) if j != labels[i]])
            silhouette_scores.append((b - a) / max(a, b))
        SC = np.mean(silhouette_scores)

        total_mean = X.mean(axis=0)
        B = sum(len(X[labels == i]) * ((centroid - total_mean) ** 2).sum() for i, centroid in enumerate(model.centroids))
        W = sum(((X[labels == i] - centroid) ** 2).sum() for i, centroid in enumerate(model.centroids))
        CH = (B / (k - 1)) / (W / (len(X) - k))

        davies_bouldin_scores = []
        for i in range(k):
            max_ratio = 0
            for j in range(k):
                if i != j:
                    s_i = np.mean(np.sqrt(((X[labels == i] - model.centroids[i]) ** 2).sum(axis=1)))
                    s_j = np.mean(np.sqrt(((X[labels == j] - model.centroids[j]) ** 2).sum(axis=1)))
                    d_ij = np.sqrt(((model.centroids[i] - model.centroids[j]) ** 2).sum())
                    max_ratio = max(max_ratio, (s_i + s_j) / d_ij)
            davies_bouldin_scores.append(max_ratio)
        DB = np.mean(davies_bouldin_scores)

        elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒

        # 写入结果
        f.write(f"k={k}, SSE={SSE:.2f}, SC={SC:.2f}, CH={CH:.2f}, DB={DB:.2f}, Time={elapsed_time:.2f}ms\n")

        # 绘图并保存
        fig, ax = plt.subplots()
        for i in range(k):
            ax.scatter(X[labels == i][:, 0], X[labels == i][:, 1], s=20, alpha=0.8, label=f'Cluster {i + 1}')
        ax.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', marker='x', s=100, linewidths=2, label='Centroids')
        ax.legend(loc='best')
        plt.title(f"KMeans Clustering (k={k})")
        plt.savefig(f"kmeans_pic/kmeans_k={k}.png", dpi=300, bbox_inches='tight')
        plt.close()

print(f"任务完成，结果已保存到 {result_file} ，图像存储在kmeans_pic文件夹中。")
