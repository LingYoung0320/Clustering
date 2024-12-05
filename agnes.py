import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import heapq

'''
AGNES层次聚类，采用自底向上聚合策略的算法。先将数据集的每个样本看做一个初始的聚类群，然后算法运行的每一步中找出距离最近的两个
群结迟行合并，该过程不断重复，直至达到预设的聚类群的个数。
'''
# 计算欧式距离矩阵
def calculate_distance_matrix(dataset):
    # 使用广播计算距离矩阵，避免手动循环
    distance_matrix = np.sqrt(np.sum((dataset[:, np.newaxis, :] - dataset[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(distance_matrix, float('inf'))
    return distance_matrix

# AGNES聚类算法实现，使用堆来加速最小距离的查找
def agnes(dataset, n_clusters):
    n_samples = len(dataset)
    clusters = [{i} for i in range(n_samples)]
    distance_matrix = calculate_distance_matrix(dataset)

    # 初始化一个最小堆来保存距离
    heap = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            heapq.heappush(heap, (distance_matrix[i, j], i, j))

    # 使用一个集合来跟踪有效的群索引
    valid_clusters = set(range(n_samples))

    while len(valid_clusters) > n_clusters:
        # 从堆中取出距离最小的两个群
        while True:
            min_dist, cluster_i, cluster_j = heapq.heappop(heap)
            if cluster_i in valid_clusters and cluster_j in valid_clusters:
                break

        # 合并群
        clusters[cluster_i] = clusters[cluster_i].union(clusters[cluster_j])
        valid_clusters.remove(cluster_j)

        # 更新距离矩阵并将新的距离加入堆中
        for i in valid_clusters:
            if i != cluster_i:
                new_dist = max(
                    np.linalg.norm(dataset[list(clusters[cluster_i])][:, np.newaxis] - dataset[list(clusters[i])], axis=2).flatten()
                ) if len(clusters[cluster_i]) > 0 and len(clusters[i]) > 0 else float('inf')
                distance_matrix[cluster_i, i] = new_dist
                distance_matrix[i, cluster_i] = new_dist
                heapq.heappush(heap, (new_dist, min(cluster_i, i), max(cluster_i, i)))

    return [clusters[i] for i in valid_clusters]

# 获取当前时间
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("\u5f53\u524d\u65f6\u95f4:", formatted_time)

# 测试代码, 使用西瓜书数据集4.0
dataset = np.loadtxt('data_test.txt')
results = agnes(dataset, 15)

# 可视化结果
color = [
    'orange', 'yellowgreen', 'olivedrab', 'darkseagreen', 'darkcyan',
    'darkturquoise', 'deepskyblue', 'steelblue', 'slategray', 'royalblue',
    'mediumpurple', 'darkmagenta', 'thistle', 'tomato', 'lightpink',
    'indigo', 'navy', 'darkslategray', 'darkred', 'dimgray'
]
centroids = []

for idx, r in enumerate(results):
    drawpoints = list(r)
    drawdata = dataset[drawpoints]
    plt.scatter(drawdata[:, 0], drawdata[:, 1], color=color[idx % len(color)], marker='o', s=20, alpha=0.8, label=f'Cluster {idx + 1}')
    # 计算质心，这里选择聚类中的几何中心
    centroid = np.mean(drawdata, axis=0)
    centroids.append(centroid)
    plt.scatter(centroid[0], centroid[1], color='r', marker='x', s=100, linewidths=2)

# 添加图例
plt.scatter([], [], color='r', marker='x', s=100, linewidths=2, label='Centroids')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()  # 自动调整布局，避免图例和图形重叠
plt.savefig('ag.png', dpi=720, bbox_inches='tight')
plt.show()

# 获取当前时间
end_time = datetime.now()
formatted_end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
print("\u7ed3\u675f\u65f6\u95f4:", formatted_end_time)
