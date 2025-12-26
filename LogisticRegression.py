import numpy as np


class LogisticRegression:
    """向量化逻辑回归"""

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, l2_reg=0.01, verbose=True):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l2 = l2_reg
        self.verbose = verbose
        self.w = None
        self.b = 0

    def _sigmoid(self, z):
        # Clip防止溢出
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # 初始化
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.max_iter):
            # 1. 线性模型 (Matrix Multiplication)
            linear = X.dot(self.w) + self.b

            # 2. 预测
            y_pred = self._sigmoid(linear)

            # 3. 梯度计算
            error = y_pred - y
            # dw = (1/m) * X.T @ error + regularization
            dw = (1 / n_samples) * np.dot(X.T, error) + (self.l2 * self.w)
            db = (1 / n_samples) * np.sum(error)

            # 4. 更新
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # 收敛检查
            if i % 100 == 0:
                loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
                if self.verbose:
                    print(f"LR Iter {i}, Loss: {loss:.4f}")
                if loss < self.tol:
                    break
        return self

    def predict_proba(self, X):
        X = np.array(X)
        linear = X.dot(self.w) + self.b
        probs = self._sigmoid(linear)
        return np.column_stack((1 - probs, probs))

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)