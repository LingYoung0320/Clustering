import os
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置随机种子
seed_value = 2023
np.random.seed(seed_value)

# 定义高斯混合聚类类
class GMM:
    def __init__(self, K, max_iter=10000, eps=1e-10):
        self.K = K  # 聚类数量
        self.max_iter = max_iter  # 最大迭代次数
        self.eps = eps  # 迭代停止的阈值

    def fit(self, X):
        self.N = X.shape[0]  # 样本数量
        self.D = X.shape[1]  # 特征数量
        self.X = X  # 样本数据

        # 初始化混合系数，均值和协方差矩阵
        self.pi = np.ones(self.K) / self.K
        self.mu = np.random.randn(self.K, self.D)
        self.sigma = np.array([np.eye(self.D)] * self.K)

        # 迭代优化
        for i in range(self.max_iter):
            gamma = self.get_gamma()
            self.update_params(gamma)
            if self.is_converged(gamma):
                break

    def get_gamma(self):
        gamma = np.zeros((self.N, self.K))
        for k in range(self.K):
            gamma[:, k] = self.pi[k] * self.multivariate_normal_pdf(self.X, self.mu[k], self.sigma[k])
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def update_params(self, gamma):
        Nk = np.sum(gamma, axis=0)
        self.pi = Nk / self.N
        self.mu = np.dot(gamma.T, self.X) / Nk.reshape(-1, 1)
        for k in range(self.K):
            X_centered = self.X - self.mu[k]
            self.sigma[k] = np.dot((X_centered.T * gamma[:, k]), X_centered) / Nk[k]
            self.sigma[k] += 1e-6 * np.eye(self.sigma.shape[1])

    def is_converged(self, gamma):
        return np.max(np.abs(gamma - self.get_gamma())) < self.eps

    def multivariate_normal_pdf(self, X, mean, cov):
        d = mean.shape[0]
        diff = X - mean
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        maha_dist = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
        normalization_factor = (2 * np.pi) ** (-d / 2) * cov_det ** (-0.5)
        return normalization_factor * np.exp(-0.5 * maha_dist)

# 创建结果保存目录
os.makedirs('gmm_pic', exist_ok=True)

# 导入数据
data = np.loadtxt('normalized_data.txt')
X = data

# 文件保存路径
result_file = 'gmm_2to20_result.txt'
with open(result_file, 'w') as file:
    file.write("K\tSSE\tSC\tCH\tDB\tTime(ms)\n")

# 遍历不同的聚类数
for k in range(2, 21):
    start_time = time.time()

    model = GMM(K=k)
    model.fit(X)

    # 获取簇所属类别
    gamma = model.get_gamma()
    cluster_label = np.argmax(gamma, axis=1)
    centroids = model.mu

    # SSE
    SSE = sum(((X - centroids[cluster_label]) ** 2).sum(axis=1))

    # SC
    silhouette_scores = []
    for i in range(len(X)):
        same_cluster = X[cluster_label == cluster_label[i]]
        other_clusters = X[cluster_label != cluster_label[i]]
        a = np.mean(np.sqrt(((same_cluster - X[i]) ** 2).sum(axis=1))) if len(same_cluster) > 1 else 0
        b = np.min([np.mean(np.sqrt(((X[cluster_label == j] - X[i]) ** 2).sum(axis=1))) for j in range(k) if j != cluster_label[i]])
        silhouette_scores.append((b - a) / max(a, b))
    SC = np.mean(silhouette_scores)

    # CH
    total_mean = X.mean(axis=0)
    B = sum(len(X[cluster_label == i]) * ((centroid - total_mean) ** 2).sum() for i, centroid in enumerate(centroids))
    W = sum(((X[cluster_label == i] - centroid) ** 2).sum() for i, centroid in enumerate(centroids))
    CH = (B / (k - 1)) / (W / (len(X) - k))

    # DB
    davies_bouldin_scores = []
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j:
                s_i = np.mean(np.sqrt(((X[cluster_label == i] - centroids[i]) ** 2).sum(axis=1)))
                s_j = np.mean(np.sqrt(((X[cluster_label == j] - centroids[j]) ** 2).sum(axis=1)))
                d_ij = np.sqrt(((centroids[i] - centroids[j]) ** 2).sum())
                max_ratio = max(max_ratio, (s_i + s_j) / d_ij)
        davies_bouldin_scores.append(max_ratio)
    DB = np.mean(davies_bouldin_scores)

    # 时间
    elapsed_time = (time.time() - start_time) * 1000

    # 写入结果
    with open(result_file, 'a') as file:
        file.write(f"{k}\t{SSE:.4f}\t{SC:.4f}\t{CH:.4f}\t{DB:.4f}\t{elapsed_time:.2f}\n")

    # 绘图
    plt.figure()
    for i in range(k):
        plt.scatter(X[cluster_label == i][:, 0], X[cluster_label == i][:, 1], s=10, alpha=0.8, label=f'Cluster {i + 1}')
    for i in range(k):
        plt.scatter(model.mu[i, 0], model.mu[i, 1], c='red', marker='x', s=50, linewidths=2)
    plt.legend()
    plt.title(f'GMM with K={k}')
    plt.tight_layout()
    plt.savefig(f'gmm_pic/gmm_k={k}.png', dpi=300)
    plt.close()

print("任务完成，结果已保存到gmm_2to20_result.txt，图像存储在gmm_pic文件夹中。")
