import numpy as np
import pandas as pd
from scipy import sparse
import math
import time
from collections import Counter, defaultdict
import heapq
import itertools
from typing import List, Tuple, Dict, Optional, Union, Callable
import random


class CrossValidation:
    """手写实现交叉验证"""

    @staticmethod
    def stratified_kfold(y, n_splits=5, shuffle=True, random_state=None):
        """
        分层K折交叉验证
        返回: 生成器，产生(train_indices, val_indices)
        """
        n_samples = len(y)

        # 获取每个类别的索引
        class_indices = {}
        for i, label in enumerate(y):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)

        # 打乱每个类别的索引
        if shuffle:
            if random_state is not None:
                random.seed(random_state)
            for label in class_indices:
                random.shuffle(class_indices[label])

        # 计算每折的类别分布
        fold_indices = [[] for _ in range(n_splits)]

        for label, indices in class_indices.items():
            n_label = len(indices)

            # 计算每折应该分到的样本数
            fold_sizes = [n_label // n_splits] * n_splits
            for i in range(n_label % n_splits):
                fold_sizes[i] += 1

            # 分配样本到各折
            start = 0
            for fold_idx in range(n_splits):
                end = start + fold_sizes[fold_idx]
                fold_indices[fold_idx].extend(indices[start:end])
                start = end

        # 打乱每折内的索引
        if shuffle:
            for fold_idx in range(n_splits):
                random.shuffle(fold_indices[fold_idx])

        # 生成训练/验证索引
        for fold_idx in range(n_splits):
            val_indices = fold_indices[fold_idx]
            train_indices = []
            for other_fold_idx in range(n_splits):
                if other_fold_idx != fold_idx:
                    train_indices.extend(fold_indices[other_fold_idx])

            yield train_indices, val_indices

    @staticmethod
    def kfold(n_samples, n_splits=5, shuffle=True, random_state=None):
        """
        普通K折交叉验证
        """
        indices = list(range(n_samples))
        if shuffle:
            if random_state is not None:
                random.seed(random_state)
            random.shuffle(indices)

        fold_size = n_samples // n_splits
        fold_sizes = [fold_size] * n_splits
        for i in range(n_samples % n_splits):
            fold_sizes[i] += 1

        start = 0
        fold_indices = []
        for fold_size in fold_sizes:
            end = start + fold_size
            fold_indices.append(indices[start:end])
            start = end

        for fold_idx in range(n_splits):
            val_indices = fold_indices[fold_idx]
            train_indices = []
            for other_fold_idx in range(n_splits):
                if other_fold_idx != fold_idx:
                    train_indices.extend(fold_indices[other_fold_idx])

            yield train_indices, val_indices

    @staticmethod
    def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
        """训练测试分割"""
        n_samples = len(X)
        indices = list(range(n_samples))

        if shuffle:
            if random_state is not None:
                random.seed(random_state)
            random.shuffle(indices)

        test_count = int(n_samples * test_size)
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

        # 分割X和y
        if isinstance(X[0], list):  # 2D数组
            X_train = [X[i] for i in train_indices]
            X_test = [X[i] for i in test_indices]
        else:  # 1D数组
            X_train = [X[i] for i in train_indices]
            X_test = [X[i] for i in test_indices]

        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def cross_val_predict(model, X, y, cv=5, method='predict_proba', verbose=True):
        """
        交叉验证预测
        返回: OOF预测
        """
        n_samples = len(X)
        if method == 'predict':
            oof_preds = [0] * n_samples
        else:  # predict_proba
            oof_preds = [[0.0, 0.0] for _ in range(n_samples)]

        fold_idx = 1
        for train_idx, val_idx in CrossValidation.stratified_kfold(y, n_splits=cv):
            if verbose:
                print(f"Training fold {fold_idx}/{cv}")

            # 分割数据
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]

            # 训练模型
            model.fit(X_train, y_train)

            # 预测
            if method == 'predict':
                val_preds = model.predict(X_val)
                for i, idx in enumerate(val_idx):
                    oof_preds[idx] = val_preds[i]
            else:  # predict_proba
                val_preds = model.predict_proba(X_val)
                for i, idx in enumerate(val_idx):
                    oof_preds[idx] = val_preds[i]

            fold_idx += 1

        return oof_preds