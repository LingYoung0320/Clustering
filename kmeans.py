import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
seed_value = 20
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

iterations = 10  # 多次迭代选取最佳模型
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
    'midnightblue', 'darkred', 'darkgreen', 'darkviolet', 'darkslategray',
    'darkolivegreen', 'darkorange', 'darkgoldenrod', 'indigo', 'deeppink',
    'darkmagenta', 'navy', 'maroon', 'olive', 'forestgreen',
    'sienna', 'slategray', 'saddlebrown', 'darkcyan', 'darkslateblue'
]

fig, ax = plt.subplots()

# 对每个类别样本进行绘制散点图
for i in range(cluster):
    ax.scatter(X[y_pred == i][:, 0],
               X[y_pred == i][:, 1],
               c=color[i],
               label=f'Cluster {i + 1}')

# 设置图例在右边
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# 调整图窗口大小
fig.set_size_inches(10, 6)

# 保存并展示图片
plt.savefig('cluster_result.png', dpi=720, bbox_inches='tight', pad_inches=0.1)
plt.show(block=True)
