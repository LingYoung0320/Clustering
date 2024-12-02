import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
mpl.use('Agg')

class GMM:
    """
    高斯混合模型（GMM），通过期望最大化（EM）算法训练。

    参数：
    ----------
    n_components : int
        高斯成分数量。

    n_iters : int
        最大迭代次数。

    tol : float
        收敛阈值，当对数似然变化小于此值时停止迭代。

    seed : int
        随机种子，用于初始化参数。
    """

    def __init__(self, n_components: int, n_iters: int, tol: float, seed: int):
        """初始化GMM模型参数"""
        self.n_components = n_components
        self.n_iters = n_iters
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        """
        训练GMM模型，使用期望最大化（EM）算法估计高斯成分参数。

        1. 初始化均值、协方差矩阵和权重。
        2. 迭代E步和M步，直到模型收敛。
        """
        n_row, n_col = X.shape
        self.resp = np.zeros((n_row, self.n_components))

        # 初始化均值、权重和协方差
        np.random.seed(self.seed)
        chosen = np.random.choice(n_row, self.n_components, replace=False)
        self.means = X[chosen]
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.covs = np.array([np.cov(X, rowvar=False)] * self.n_components)

        log_likelihood = 0
        self.converged = False
        self.log_likelihood_trace = []

        # 迭代EM算法
        for i in range(self.n_iters):
            log_likelihood_new = self._do_estep(X)  # E步
            self._do_mstep(X)  # M步

            if abs(log_likelihood_new - log_likelihood) <= self.tol:  # 收敛条件
                self.converged = True
                break

            log_likelihood = log_likelihood_new
            self.log_likelihood_trace.append(log_likelihood)

        return self

    def _do_estep(self, X):
        """
        E步：计算每个数据点属于每个高斯成分的责任，并更新责任矩阵。
        """
        self._compute_log_likelihood(X)
        log_likelihood = np.sum(np.log(np.sum(self.resp, axis=1)))
        self.resp = self.resp / self.resp.sum(axis=1, keepdims=1)  # 归一化
        return log_likelihood

    def _compute_log_likelihood(self, X):
        """
        计算每个数据点的对数似然。
        """
        for k in range(self.n_components):
            prior = self.weights[k]
            likelihood = self.multivariate_normal_pdf(X, self.means[k], self.covs[k])
            self.resp[:, k] = prior * likelihood

        return self

    def _do_mstep(self, X):
        """
        M步：根据E步结果更新模型参数（均值、协方差和权重）。
        """
        resp_weights = self.resp.sum(axis=0)  # 每个高斯成分的总责任
        self.weights = resp_weights / X.shape[0]

        weighted_sum = np.dot(self.resp.T, X)  # 更新均值
        self.means = weighted_sum / resp_weights.reshape(-1, 1)

        for k in range(self.n_components):
            diff = (X - self.means[k]).T
            weighted_sum = np.dot(self.resp[:, k] * diff, diff.T)  # 更新协方差
            self.covs[k] = weighted_sum / resp_weights[k]

        return self

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


def plot_contours(data, means, covs, title, gmm_model):
    """
    可视化高斯混合模型的结果，显示数据点和高斯分布的等高线。
    """
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')  # 绘制数据点

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    # 生成足够颜色
    cmap = ListedColormap(plt.cm.tab20.colors[:k])

    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = gmm_model.multivariate_normal_pdf(coordinates, mean, cov).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors=[cmap(i)])  # 绘制等高线

    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.savefig("test1_2.png")


def print_gmm_results(gmm_model):
    """
    打印GMM模型的训练结果：均值、协方差、权重和数据点分类。
    """
    print("GMM Model Results:")

    # 输出均值
    print("\nMeans of each Gaussian component:")
    for i, mean in enumerate(gmm_model.means):
        print(f"Component {i + 1}: {mean}")

    # 输出协方差矩阵
    print("\nCovariances of each Gaussian component:")
    for i, cov in enumerate(gmm_model.covs):
        print(f"Component {i + 1} covariance matrix:\n{cov}")

    # 输出权重
    print("\nWeights of each Gaussian component:")
    for i, weight in enumerate(gmm_model.weights):
        print(f"Component {i + 1}: {weight}")

    # 输出数据点分类
    print("\nCluster assignments (most likely component for each data point):")
    cluster_assignments = np.argmax(gmm_model.resp, axis=1)
    print(cluster_assignments)


# 主逻辑：加载数据、训练模型、打印结果、可视化
X = np.loadtxt('data2.txt')  # 读取数据
gmm = GMM(n_components=15, n_iters=1, tol=1e-4, seed=4)  # 初始化GMM
gmm.fit(X)  # 训练GMM

print_gmm_results(gmm)  # 输出训练结果
plot_contours(X, gmm.means, gmm.covs, 'Initial clusters', gmm)  # 可视化结果
