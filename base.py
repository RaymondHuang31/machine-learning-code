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


class NumpyFunctions:

    @staticmethod
    def mean(arr, axis=None):
        if axis is None:
            return sum(arr) / len(arr) if len(arr) > 0 else 0
        elif axis == 0:
            return [sum(col) / len(col) for col in zip(*arr)]
        elif axis == 1:
            return [sum(row) / len(row) for row in arr]

    @staticmethod
    def std(arr, axis=None, ddof=1):
        if axis is None:
            mean_val = NumpyFunctions.mean(arr)
            variance = sum((x - mean_val) ** 2 for x in arr) / (len(arr) - ddof)
            return math.sqrt(variance)
        elif axis == 0:
            result = []
            for col in zip(*arr):
                mean_val = NumpyFunctions.mean(col)
                variance = sum((x - mean_val) ** 2 for x in col) / (len(col) - ddof)
                result.append(math.sqrt(variance))
            return result
        elif axis == 1:
            result = []
            for row in arr:
                mean_val = NumpyFunctions.mean(row)
                variance = sum((x - mean_val) ** 2 for x in row) / (len(row) - ddof)
                result.append(math.sqrt(variance))
            return result

    @staticmethod
    def concatenate(arrays, axis=0):
        if axis == 0:
            return [item for arr in arrays for item in arr]
        elif axis == 1:
            return [list(itertools.chain(*row)) for row in zip(*arrays)]

    @staticmethod
    def zeros(shape):
        if isinstance(shape, int):
            return [0.0] * shape
        elif len(shape) == 1:
            return [0.0] * shape[0]
        elif len(shape) == 2:
            return [[0.0] * shape[1] for _ in range(shape[0])]

    @staticmethod
    def ones(shape):
        if isinstance(shape, int):
            return [1.0] * shape
        elif len(shape) == 1:
            return [1.0] * shape[0]
        elif len(shape) == 2:
            return [[1.0] * shape[1] for _ in range(shape[0])]

    @staticmethod
    def argmax(arr):
        max_val = arr[0]
        max_idx = 0
        for i, val in enumerate(arr):
            if val > max_val:
                max_val = val
                max_idx = i
        return max_idx

    @staticmethod
    def argsort(arr, reverse=False):
        indexed = list(enumerate(arr))
        indexed.sort(key=lambda x: x[1], reverse=reverse)
        return [i for i, _ in indexed]

    @staticmethod
    def matmul(A, B):
        m, n = len(A), len(A[0])
        p = len(B[0])
        result = [[0.0] * p for _ in range(m)]
        for i in range(m):
            for k in range(n):
                if A[i][k] != 0:
                    for j in range(p):
                        result[i][j] += A[i][k] * B[k][j]
        return result

    @staticmethod
    def transpose(matrix):
        if not matrix:
            return []
        n = len(matrix)
        m = len(matrix[0])
        result = [[0.0] * n for _ in range(m)]
        for i in range(n):
            for j in range(m):
                result[j][i] = matrix[i][j]
        return result
