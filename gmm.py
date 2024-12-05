import numpy as np
import matplotlib.pyplot as plt

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
            # E 步
            gamma = self.get_gamma()

            # M 步
            self.update_params(gamma)

            # 判断迭代是否终止
            if self.is_converged(gamma):
                break

    def get_gamma(self):
        # 计算后验概率，即每个样本属于每个聚类的概率
        gamma = np.zeros((self.N, self.K))
        for k in range(self.K):
            gamma[:, k] = self.pi[k] * self.multivariate_normal_pdf(self.X, self.mu[k], self.sigma[k])
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def update_params(self, gamma):
        # 更新混合系数，均值和协方差矩阵
        Nk = np.sum(gamma, axis=0)
        self.pi = Nk / self.N
        self.mu = np.dot(gamma.T, self.X) / Nk.reshape(-1, 1)
        for k in range(self.K):
            X_centered = self.X - self.mu[k]
            self.sigma[k] = np.dot((X_centered.T * gamma[:, k]), X_centered) / Nk[k]
            self.sigma[k] += 1e-6 * np.eye(self.sigma.shape[1])

    def is_converged(self, gamma):
        # 判断是否满足迭代停止的阈值
        return np.max(np.abs(gamma - self.get_gamma())) < self.eps

    def multivariate_normal_pdf(self, X, mean, cov):
        """
        计算多维正态分布的概率密度函数（PDF）。
        """
        d = mean.shape[0]
        diff = X - mean
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        maha_dist = np.sum(np.dot(diff, cov_inv) * diff, axis=1)  # 马哈拉诺比斯距离
        normalization_factor = (2 * np.pi) ** (-d / 2) * cov_det ** (-0.5)
        pdf_values = normalization_factor * np.exp(-0.5 * maha_dist)
        return pdf_values

# 导入数据
data = np.loadtxt('normalized_data.txt')  # 使用 NumPy 读取数据
X = data  # 取出所有数据

# 训练高斯混合聚类模型
cluster = 15
model = GMM(K=cluster)
model.fit(X)

# 获取簇所属类别
gamma = model.get_gamma()
cluster_label = np.argmax(gamma, axis=1)

# 计算簇的质心
centroids = model.mu

# 计算误差平方和 (SSE)
SSE = sum(((X - centroids[cluster_label]) ** 2).sum(axis=1))
print(f"Sum of Squared Errors (SSE): {SSE}")

# 计算轮廓系数 (Silhouette Coefficient, SC)
silhouette_scores = []
for i in range(len(X)):
    same_cluster = X[cluster_label == cluster_label[i]]
    other_clusters = X[cluster_label != cluster_label[i]]
    a = np.mean(np.sqrt(((same_cluster - X[i]) ** 2).sum(axis=1))) if len(same_cluster) > 1 else 0
    b = np.min([np.mean(np.sqrt(((X[cluster_label == j] - X[i]) ** 2).sum(axis=1))) for j in range(cluster) if j != cluster_label[i]])
    silhouette_scores.append((b - a) / max(a, b))
SC = np.mean(silhouette_scores)
print(f"Silhouette Coefficient (SC): {SC}")

# 计算 Calinski-Harabasz 指标 (CH)
total_mean = X.mean(axis=0)
B = sum(len(X[cluster_label == i]) * ((centroid - total_mean) ** 2).sum() for i, centroid in enumerate(centroids))
W = sum(((X[cluster_label == i] - centroid) ** 2).sum() for i, centroid in enumerate(centroids))
CH = (B / (cluster - 1)) / (W / (len(X) - cluster))
print(f"Calinski-Harabasz (CH): {CH}")

# 计算 Davies-Bouldin 指标 (DB)
davies_bouldin_scores = []
for i in range(cluster):
    max_ratio = 0
    for j in range(cluster):
        if i != j:
            s_i = np.mean(np.sqrt(((X[cluster_label == i] - centroids[i]) ** 2).sum(axis=1)))
            s_j = np.mean(np.sqrt(((X[cluster_label == j] - centroids[j]) ** 2).sum(axis=1)))
            d_ij = np.sqrt(((centroids[i] - centroids[j]) ** 2).sum())
            max_ratio = max(max_ratio, (s_i + s_j) / d_ij)
    davies_bouldin_scores.append(max_ratio)
DB = np.mean(davies_bouldin_scores)
print(f"Davies-Bouldin (DB): {DB}")

# 聚类结果可视化
# 绘制颜色
color = [
    'orange', 'yellowgreen', 'olivedrab', 'darkseagreen', 'darkcyan',
    'darkturquoise', 'deepskyblue', 'steelblue', 'slategray', 'royalblue',
    'mediumpurple', 'darkmagenta', 'thistle', 'tomato', 'lightpink',
    'indigo', 'navy', 'darkslategray', 'darkred', 'dimgray'
]

# 绘制每个类别样本的散点图并添加图例
for i in range(cluster):
    plt.scatter(X[cluster_label == i][:, 0],
                X[cluster_label == i][:, 1],
                c=color[i], s=20, alpha=0.8,
                label=f'Cluster {i + 1}')

# 绘制每个聚类的质心（用红色叉叉）
for i in range(cluster):
    plt.scatter(model.mu[i, 0], model.mu[i, 1], c='red', marker='x', s=100, linewidths=2)

# 将图例放置在图的右边
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[plt.Line2D([0], [0], marker='x', color='red', label='Centroid', markersize=10, linestyle='')] + [plt.Line2D([0], [0], marker='o', color=color[i], label=f'Cluster {i + 1}', markersize=5, linestyle='') for i in range(cluster)])
plt.tight_layout()  # 自动调整布局，避免图例和图形重叠

# 保存并展示图形
plt.savefig('gmm.png', dpi=720, bbox_inches='tight')
plt.show()
