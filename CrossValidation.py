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

    @staticmethod
    def stratified_kfold(y, n_splits=5, shuffle=True, random_state=None):
        n_samples = len(y)

        class_indices = {}
        for i, label in enumerate(y):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)

        if shuffle:
            if random_state is not None:
                random.seed(random_state)
            for label in class_indices:
                random.shuffle(class_indices[label])

        fold_indices = [[] for _ in range(n_splits)]

        for label, indices in class_indices.items():
            n_label = len(indices)

            fold_sizes = [n_label // n_splits] * n_splits
            for i in range(n_label % n_splits):
                fold_sizes[i] += 1

            start = 0
            for fold_idx in range(n_splits):
                end = start + fold_sizes[fold_idx]
                fold_indices[fold_idx].extend(indices[start:end])
                start = end

        if shuffle:
            for fold_idx in range(n_splits):
                random.shuffle(fold_indices[fold_idx])

        for fold_idx in range(n_splits):
            val_indices = fold_indices[fold_idx]
            train_indices = []
            for other_fold_idx in range(n_splits):
                if other_fold_idx != fold_idx:
                    train_indices.extend(fold_indices[other_fold_idx])

            yield train_indices, val_indices

    @staticmethod
    def kfold(n_samples, n_splits=5, shuffle=True, random_state=None):
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
        n_samples = len(X)
        indices = list(range(n_samples))

        if shuffle:
            if random_state is not None:
                random.seed(random_state)
            random.shuffle(indices)

        test_count = int(n_samples * test_size)
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

        if isinstance(X[0], list):
            X_train = [X[i] for i in train_indices]
            X_test = [X[i] for i in test_indices]
        else:
            X_train = [X[i] for i in train_indices]
            X_test = [X[i] for i in test_indices]

        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def cross_val_predict(model, X, y, cv=5, method='predict_proba', verbose=True):
        n_samples = len(X)
        if method == 'predict':
            oof_preds = [0] * n_samples
        else:
            oof_preds = [[0.0, 0.0] for _ in range(n_samples)]

        fold_idx = 1
        for train_idx, val_idx in CrossValidation.stratified_kfold(y, n_splits=cv):
            if verbose:
                print(f"Training fold {fold_idx}/{cv}")

            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]

            model.fit(X_train, y_train)

            if method == 'predict':
                val_preds = model.predict(X_val)
                for i, idx in enumerate(val_idx):
                    oof_preds[idx] = val_preds[i]
            else:
                val_preds = model.predict_proba(X_val)
                for i, idx in enumerate(val_idx):
                    oof_preds[idx] = val_preds[i]

            fold_idx += 1

        return oof_preds
