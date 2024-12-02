import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
mpl.use('Agg')

class GMM:
    """
    Full covariance Gaussian Mixture Model,
    trained using Expectation Maximization.

    Parameters
    ----------
    n_components : int
        Number of clusters/mixture components in which the data will be
        partitioned into.

    n_iters : int
        Maximum number of iterations to run the algorithm.

    tol : float
        Tolerance. If the log-likelihood between two iterations is smaller than
        the specified tolerance level, the algorithm will stop performing the
        EM optimization.

    seed : int
        Seed / random state used to initialize the parameters.
    """

    def __init__(self, n_components: int, n_iters: int, tol: float, seed: int):
        self.n_components = n_components
        self.n_iters = n_iters
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        # data's dimensionality and responsibility vector
        n_row, n_col = X.shape
        self.resp = np.zeros((n_row, self.n_components))

        # initialize parameters
        np.random.seed(self.seed)
        chosen = np.random.choice(n_row, self.n_components, replace=False)
        self.means = X[chosen]
        self.weights = np.full(self.n_components, 1 / self.n_components)

        # covariance matrices initialized to the sample covariance of X
        shape = self.n_components, n_col, n_col
        self.covs = np.array([np.cov(X, rowvar=False)] * self.n_components)

        log_likelihood = 0
        self.converged = False
        self.log_likelihood_trace = []

        for i in range(self.n_iters):
            log_likelihood_new = self._do_estep(X)
            self._do_mstep(X)

            if abs(log_likelihood_new - log_likelihood) <= self.tol:
                self.converged = True
                break

            log_likelihood = log_likelihood_new
            self.log_likelihood_trace.append(log_likelihood)

        return self

    def _do_estep(self, X):
        """
        E-step: compute responsibilities,
        update resp matrix so that resp[j, k] is the responsibility of cluster k for data point j,
        to compute likelihood of seeing data point j given cluster k, use custom multivariate normal PDF
        """
        self._compute_log_likelihood(X)
        log_likelihood = np.sum(np.log(np.sum(self.resp, axis=1)))

        # normalize over all possible cluster assignments
        self.resp = self.resp / self.resp.sum(axis=1, keepdims=1)
        return log_likelihood

    def _compute_log_likelihood(self, X):
        for k in range(self.n_components):
            prior = self.weights[k]
            likelihood = self.multivariate_normal_pdf(X, self.means[k], self.covs[k])
            self.resp[:, k] = prior * likelihood

        return self

    def _do_mstep(self, X):
        """M-step, update parameters"""

        # total responsibility assigned to each cluster, N^{soft}
        resp_weights = self.resp.sum(axis=0)

        # weights
        self.weights = resp_weights / X.shape[0]

        # means
        weighted_sum = np.dot(self.resp.T, X)
        self.means = weighted_sum / resp_weights.reshape(-1, 1)
        # covariance
        for k in range(self.n_components):
            diff = (X - self.means[k]).T
            weighted_sum = np.dot(self.resp[:, k] * diff, diff.T)
            self.covs[k] = weighted_sum / resp_weights[k]

        return self

    def multivariate_normal_pdf(self, X, mean, cov):
        """
        Compute the multivariate normal PDF for data points X with mean and covariance matrix.
        """
        d = mean.shape[0]  # dimensionality
        diff = X - mean
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)

        # Calculating the Mahalanobis distance term
        maha_dist = np.sum(np.dot(diff, cov_inv) * diff, axis=1)

        # PDF formula
        normalization_factor = (2 * np.pi) ** (-d / 2) * cov_det ** (-0.5)
        pdf_values = normalization_factor * np.exp(-0.5 * maha_dist)
        return pdf_values

def plot_contours(data, means, covs, title, gmm_model):
    """visualize the gaussian components over the data"""
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    # 使用 ListedColormap 动态生成足够的颜色
    cmap = ListedColormap(plt.cm.tab20.colors[:k])  # 通过 tab20 获取前 k 种颜色

    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = gmm_model.multivariate_normal_pdf(coordinates, mean, cov).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors=[cmap(i)])

    plt.title(title)
    plt.tight_layout()

    plt.show()
    plt.savefig("test1_2.png")


# 读取 CSV 文件
df = pd.read_csv('data2.csv', header=None)  # 如果没有列名，可以设置header=None
# 假设数据的第一列是 x，第二列是 y
X = df.to_numpy()  # 转换为 NumPy 数组
df = df.apply(pd.to_numeric, errors='coerce')  # 强制转换为数值型，无法转换的会变为 NaN
df = df.dropna()  # 删除任何包含 NaN 的行

# 将数据转化为 NumPy 数组
X = df.to_numpy()

# 初始化并训练 GMM 模型
gmm = GMM(n_components=15, n_iters=1, tol=1e-4, seed=4)
gmm.fit(X)


def print_gmm_results(gmm_model):
    """Print the GMM model results (means, covariances, weights, and responsibilities)."""
    print("GMM Model Results:")

    # 输出每个高斯组件的均值
    print("\nMeans of each Gaussian component:")
    for i, mean in enumerate(gmm_model.means):
        print(f"Component {i + 1}: {mean}")

    # 输出每个高斯组件的协方差矩阵
    print("\nCovariances of each Gaussian component:")
    for i, cov in enumerate(gmm_model.covs):
        print(f"Component {i + 1} covariance matrix:\n{cov}")

    # 输出每个高斯组件的权重
    print("\nWeights of each Gaussian component:")
    for i, weight in enumerate(gmm_model.weights):
        print(f"Component {i + 1}: {weight}")

    # 输出每个数据点的最大责任（即每个数据点属于哪个高斯组件）
    print("\nCluster assignments (most likely component for each data point):")
    cluster_assignments = np.argmax(gmm_model.resp, axis=1)
    print(cluster_assignments)


# 调用函数输出 GMM 结果
print_gmm_results(gmm)

# 可视化结果
plot_contours(X, gmm.means, gmm.covs, 'Initial clusters', gmm)
