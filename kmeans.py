import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
seed_value = 2030
np.random.seed(seed_value)

class KMeans:
    def __init__(self, k=3, max_iter=100, tol=0.0001):
        self.k = k  # 聚类的数量
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛容忍度

    def fit(self, X):
        # 随机初始化聚类中心
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False), :]

        # 迭代聚类过程
        for i in range(self.max_iter):
            # 计算每个点到聚类中心的距离
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))

            # 分配每个点到距离最近的聚类中心
            labels = distances.argmin(axis=0)

            # 计算新的聚类中心
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])

            # 计算收敛误差
            delta = np.abs(self.centroids - new_centroids).max()

            # 如果收敛误差小于容忍度，退出迭代
            if delta < self.tol:
                break

            # 更新聚类中心
            self.centroids = new_centroids

    def predict(self, X):
        # 计算每个点到聚类中心的距离
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))

        # 分配每个点到距离最近的聚类中心
        labels = distances.argmin(axis=0)

        return labels


# 导入数据
# 修改为从txt文件读取数据
data = np.loadtxt("normalized_data.txt", delimiter=" ")

# 确保特征为二维数据
X = data[:, :2]

# 选取最优的聚类结果
best_inertia = float('inf')
best_model = None
best_labels = None

iterations = 100  # 多次迭代选取最佳模型
cluster = 15  # 根据需求更改聚类数量

for _ in range(iterations):
    model = KMeans(k=cluster)
    model.fit(X)
    labels = model.predict(X)

    # 计算当前聚类的总误差平方和 (inertia)
    inertia = sum(np.min(np.sqrt(((X - model.centroids[:, np.newaxis]) ** 2).sum(axis=2)), axis=0))

    # 选取最优的模型
    if inertia < best_inertia:
        best_inertia = inertia
        best_model = model
        best_labels = labels

# 使用最优模型的聚类结果
y_pred = best_labels

# 聚类结果可视化
# 绘制颜色
color = [
    'orange', 'yellowgreen', 'olivedrab', 'darkseagreen', 'darkcyan',
    'darkturquoise', 'deepskyblue', 'steelblue', 'slategray', 'royalblue',
    'mediumpurple', 'darkmagenta', 'thistle', 'tomato', 'lightpink',
    'indigo', 'navy', 'darkslategray', 'darkred', 'dimgray'
]

fig, ax = plt.subplots()

# 对每个类别样本进行绘制散点图
for i in range(cluster):
    ax.scatter(X[y_pred == i][:, 0],
               X[y_pred == i][:, 1],
               c=color[i], s=20, alpha=0.8,
               label=f'Cluster {i + 1}')

# 绘制质心
ax.scatter(best_model.centroids[:, 0], best_model.centroids[:, 1],
           c='red', marker='x', s=100, linewidths=2, label='Centroids')

# 设置图例在右边
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# 调整图窗口大小
fig.set_size_inches(8, 6)
plt.tight_layout()  # 自动调整布局，避免图例和图形重叠
# 保存并展示图片
plt.savefig('cluster_result.png', dpi=720, bbox_inches='tight', pad_inches=0.1)
plt.show(block=True)

# 计算聚类指标
# 误差平方和 (Sum of Squared Errors, SSE)
SSE = sum(((X - best_model.centroids[best_labels]) ** 2).sum(axis=1))
print(f"Sum of Squared Errors (SSE): {SSE}")

# 轮廓系数 (Silhouette Coefficient, SC)
silhouette_scores = []
for i in range(len(X)):
    same_cluster = X[best_labels == best_labels[i]]
    other_clusters = X[best_labels != best_labels[i]]
    a = np.mean(np.sqrt(((same_cluster - X[i]) ** 2).sum(axis=1))) if len(same_cluster) > 1 else 0
    b = np.min([np.mean(np.sqrt(((X[best_labels == j] - X[i]) ** 2).sum(axis=1))) for j in range(cluster) if j != best_labels[i]])
    silhouette_scores.append((b - a) / max(a, b))
SC = np.mean(silhouette_scores)
print(f"Silhouette Coefficient (SC): {SC}")

# Calinski-Harabasz (CH)
total_mean = X.mean(axis=0)
B = sum(len(X[best_labels == i]) * ((centroid - total_mean) ** 2).sum() for i, centroid in enumerate(best_model.centroids))
W = sum(((X[best_labels == i] - centroid) ** 2).sum() for i, centroid in enumerate(best_model.centroids))
CH = (B / (cluster - 1)) / (W / (len(X) - cluster))
print(f"Calinski-Harabasz (CH): {CH}")

# Davies-Bouldin (DB)
davies_bouldin_scores = []
for i in range(cluster):
    max_ratio = 0
    for j in range(cluster):
        if i != j:
            s_i = np.mean(np.sqrt(((X[best_labels == i] - best_model.centroids[i]) ** 2).sum(axis=1)))
            s_j = np.mean(np.sqrt(((X[best_labels == j] - best_model.centroids[j]) ** 2).sum(axis=1)))
            d_ij = np.sqrt(((best_model.centroids[i] - best_model.centroids[j]) ** 2).sum())
            max_ratio = max(max_ratio, (s_i + s_j) / d_ij)
    davies_bouldin_scores.append(max_ratio)
DB = np.mean(davies_bouldin_scores)
print(f"Davies-Bouldin (DB): {DB}")
