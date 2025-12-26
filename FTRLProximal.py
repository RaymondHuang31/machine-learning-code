import numpy as np
import math


class FTRLProximal:
    """FTRL-Proximal 算法"""

    def __init__(self, alpha=0.1, beta=1.0, l1=1.0, l2=1.0, dim=None):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.dim = dim
        self.z = None
        self.n = None

        if dim:
            self._initialize(dim)

    def _initialize(self, dim):
        self.dim = dim
        self.z = np.zeros(dim)
        self.n = np.zeros(dim)

    def _sigmoid(self, x):
        x = np.clip(x, -35, 35)  # 防止溢出
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y, epochs=1, verbose=True):
        # 确保是numpy数组
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        history = []  # <--- 新增

        if self.z is None:
            self._initialize(n_features)

        # FTRL通常是Online的（逐行更新），全向量化会变成不同的算法
        # 这里我们保持逻辑不变，但使用Numpy加速内部计算

        for epoch in range(epochs):
            loss_sum = 0.0

            # 使用Numpy迭代比纯Python List快
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]

                # 1. 计算权重 w
                # w = (sign(z) * l1 - z) / ((beta + sqrt(n)) / alpha + l2)
                # if |z| <= l1, w = 0

                sign_z = np.sign(self.z)
                mask = np.abs(self.z) > self.l1

                w = np.zeros(self.dim)
                if np.any(mask):
                    denom = (self.beta + np.sqrt(self.n[mask])) / self.alpha + self.l2
                    w[mask] = (sign_z[mask] * self.l1 - self.z[mask]) / denom

                # 2. 预测
                dot = np.dot(w, xi)
                p = self._sigmoid(dot)

                # 3. 梯度 g = p - y
                g = p - yi

                # 4. 更新 z 和 n
                # n_new = n + g^2 * xi^2
                # sigma = (sqrt(n_new) - sqrt(n)) / alpha
                # z_new = z + g * xi - sigma * w

                # 仅更新非零特征以加速（对于稀疏数据）
                # 但这里的xi可能是稠密的，直接向量化更新
                n_new = self.n + (g * xi) ** 2
                sigma = (np.sqrt(n_new) - np.sqrt(self.n)) / self.alpha
                self.z += g * xi - sigma * w
                self.n = n_new

                # 简单Loss记录
                loss_sum -= (yi * np.log(p + 1e-15) + (1 - yi) * np.log(1 - p + 1e-15))

            # <--- 新增
            avg_loss = loss_sum / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"FTRL Epoch {epoch + 1}, Avg Loss: {loss_sum / n_samples:.4f}")

        return history

    def predict_proba(self, X):
        X = np.array(X)
        n_samples = len(X)
        preds = []

        # 计算最终权重 w
        mask = np.abs(self.z) > self.l1
        w = np.zeros(self.dim)
        if np.any(mask):
            denom = (self.beta + np.sqrt(self.n[mask])) / self.alpha + self.l2
            w[mask] = (np.sign(self.z[mask]) * self.l1 - self.z[mask]) / denom

        # 批量预测
        dots = X.dot(w)
        probs = self._sigmoid(dots)

        return np.column_stack((1 - probs, probs))