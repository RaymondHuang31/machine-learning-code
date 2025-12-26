import numpy as np
import random
from collections import defaultdict  # 【修复点1】必须导入 defaultdict


class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

    def predict(self, x):
        if self.value is not None:
            return self.value
        if x[self.feature_idx] < self.threshold:
            return self.left.predict(x)
        return self.right.predict(x)


class DecisionTree:
    """向量化 CART 树，使用分位数近似分割"""

    def __init__(self, max_depth=5, min_samples_split=10, min_impurity=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.root = None
        # 【修复点2】初始化重要性字典，否则画图时这里是空的
        self.feature_importances_ = defaultdict(float)

    def _gini(self, y):
        m = len(y)
        if m == 0: return 0
        p = np.sum(y) / m
        return 1.0 - (p ** 2 + (1 - p) ** 2)

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1: return None, None, 0.0

        current_impurity = self._gini(y)  # 这里默认叫 impurity 比较通用
        best_gain = 0.0
        best_feature = None
        best_thr = None

        # 随机特征采样
        feature_indices = np.random.choice(n, min(n, int(np.sqrt(n)) + 10), replace=False)

        for feat_idx in feature_indices:
            vals = X[:, feat_idx]
            unique_vals = np.unique(vals)
            if len(unique_vals) > 10:
                thresholds = np.percentile(vals, np.linspace(10, 90, 8))
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

            for thr in thresholds:
                left_mask = vals < thr
                if not np.any(left_mask) or np.all(left_mask):
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]

                # 【修复点3】必须计算 Gain (父节点不纯度 - 子节点加权不纯度)
                imp_left = self._gini(y_left)
                imp_right = self._gini(y_right)
                weighted_imp = (len(y_left) * imp_left + len(y_right) * imp_right) / m
                gain = current_impurity - weighted_imp

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_thr = thr

        return best_feature, best_thr, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_labels = len(np.unique(y)) if len(y) > 0 else 0

        # 停止条件
        if (depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split):
            leaf_value = np.mean(y)
            return DecisionTreeNode(value=leaf_value)

        feat, thr, gain = self._best_split(X, y)

        if feat is None or gain < self.min_impurity:
            return DecisionTreeNode(value=np.mean(y))

        # 【修复点4】核心！将增益累加到特征重要性字典中
        # 之前代码可能漏了这一行，导致字典一直是空的
        self.feature_importances_[feat] += gain * n_samples

        left_mask = X[:, feat] < thr
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        left_node = self._build_tree(X_left, y_left, depth + 1)
        right_node = self._build_tree(X_right, y_right, depth + 1)

        return DecisionTreeNode(feature_idx=feat, threshold=thr, left=left_node, right=right_node)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.feature_importances_ = defaultdict(float)  # 每次fit前重置
        self.root = self._build_tree(X, y)
        return self

    def predict(self, X):
        X = np.array(X)
        return [self.root.predict(x) for x in X]


class RegressionTree(DecisionTree):
    def _gini(self, y):
        # 【修复点5】回归树使用方差 (Variance) 计算不纯度
        # 虽然父类方法名叫 _gini，但在回归树里它通过覆盖实现了方差计算
        if len(y) == 0: return 0
        return np.var(y)


class GBTClassifier:
    """向量化梯度提升树"""

    def __init__(self, n_estimators=20, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = 0.0
        # 【修复点6】GBT 自己的特征重要性汇总
        self.feature_importances_ = defaultdict(float)

    def _sigmoid(self, z):
        z = np.clip(z, -100, 100)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        pos = np.sum(y)
        neg = len(y) - pos
        self.init_pred = np.log((pos + 1e-5) / (neg + 1e-5))
        curr_logits = np.full(len(y), self.init_pred)

        self.feature_importances_ = defaultdict(float)  # 重置
        history = []  # 【新增】用于记录每一轮的 Loss

        for i in range(self.n_estimators):
            p = self._sigmoid(curr_logits)
            gradients = y - p

            # 拟合残差
            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=20)
            tree.fit(X, gradients)
            self.trees.append(tree)

            # 【修复点7】将子树的重要性累加到 GBT 总表里
            # 如果没有这一步，Core.py 里读取 gbt.feature_importances_ 就会是空的
            for feat, importance in tree.feature_importances_.items():
                self.feature_importances_[feat] += importance

            update = np.array(tree.predict(X))
            curr_logits += self.lr * update

            # 【新增】计算当前 Log Loss 并记录
            # LogLoss = -mean(y*log(p) + (1-y)*log(1-p))
            p_current = self._sigmoid(curr_logits)
            epsilon = 1e-15
            p_current = np.clip(p_current, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(p_current) + (1 - y) * np.log(1 - p_current))
            history.append(loss)

        return history  # 【修改】返回 history 而不是 self

    def predict_proba(self, X):
        X = np.array(X)
        curr_logits = np.full(len(X), self.init_pred)

        for tree in self.trees:
            curr_logits += self.lr * np.array(tree.predict(X))

        probs = self._sigmoid(curr_logits)
        return np.column_stack((1 - probs, probs))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)