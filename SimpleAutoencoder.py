import numpy as np


class SimpleAutoencoder:
    """向量化自编码器"""

    def __init__(self, input_dim, encoding_dim=10, learning_rate=0.01):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.lr = learning_rate

        # Xavier初始化
        limit = np.sqrt(6 / (input_dim + encoding_dim))
        self.W1 = np.random.uniform(-limit, limit, (input_dim, encoding_dim))
        self.b1 = np.zeros(encoding_dim)
        self.W2 = np.random.uniform(-limit, limit, (encoding_dim, input_dim))
        self.b2 = np.zeros(input_dim)

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        x = np.clip(x, -250, 250)
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, epochs=20, batch_size=64, verbose=False):
        X = np.array(X)
        n_samples = len(X)
        history = []

        for epoch in range(epochs):
            # Shuffle
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]

            epoch_loss = 0

            for start_idx in range(0, n_samples, batch_size):
                batch_X = X[start_idx:start_idx + batch_size]
                m = len(batch_X)

                # --- 前向传播 ---
                # Hidden: (m, enc)
                hidden = self._relu(np.dot(batch_X, self.W1) + self.b1)
                # Output: (m, in)
                output = self._sigmoid(np.dot(hidden, self.W2) + self.b2)

                # --- 计算梯度 ---
                diff = output - batch_X
                loss = np.mean(diff ** 2)
                epoch_loss += loss

                # 输出层梯度 (Sigmoid导数: out * (1-out))
                d_out = diff * (output * (1 - output))

                # W2, b2 梯度
                d_W2 = np.dot(hidden.T, d_out) / m
                d_b2 = np.sum(d_out, axis=0) / m

                # 隐藏层梯度 (ReLU导数)
                d_hidden = np.dot(d_out, self.W2.T)
                d_hidden[hidden <= 0] = 0

                # W1, b1 梯度
                d_W1 = np.dot(batch_X.T, d_hidden) / m
                d_b1 = np.sum(d_hidden, axis=0) / m

                # --- 更新 ---
                self.W1 -= self.lr * d_W1
                self.b1 -= self.lr * d_b1
                self.W2 -= self.lr * d_W2
                self.b2 -= self.lr * d_b2
            # 记录本轮平均 loss
            avg_loss = epoch_loss / (n_samples / batch_size)
            history.append(avg_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"AE Epoch {epoch + 1}, Loss: {epoch_loss / (n_samples / batch_size):.4f}")

        return history

    def encode(self, X):
        X = np.array(X)
        return self._relu(np.dot(X, self.W1) + self.b1)

    def reconstruction_error(self, X):
        X = np.array(X)
        hidden = self._relu(np.dot(X, self.W1) + self.b1)
        output = self._sigmoid(np.dot(hidden, self.W2) + self.b2)
        # 按样本计算MSE
        return np.mean((X - output) ** 2, axis=1)