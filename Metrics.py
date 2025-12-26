import numpy as np
import math
from base import NumpyFunctions

class Metrics:
    """手写实现评估指标"""

    @staticmethod
    def matthews_corrcoef(y_true, y_pred):
        """手写实现MCC计算"""
        tp = tn = fp = fn = 0
        for true, pred in zip(y_true, y_pred):
            if true == 1 and pred == 1: tp += 1
            elif true == 0 and pred == 0: tn += 1
            elif true == 0 and pred == 1: fp += 1
            elif true == 1 and pred == 0: fn += 1

        numerator = tp * tn - fp * fn
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator != 0 else 0

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """混淆矩阵"""
        tp = tn = fp = fn = 0
        for true, pred in zip(y_true, y_pred):
            if true == 1 and pred == 1: tp += 1
            elif true == 0 and pred == 0: tn += 1
            elif true == 0 and pred == 1: fp += 1
            elif true == 1 and pred == 0: fn += 1
        return [[tn, fp], [fn, tp]]

    @staticmethod
    def roc_curve(y_true, y_scores):
        """ROC曲线计算 - 严格返回 3 个值"""
        # 按分数排序
        sorted_indices = NumpyFunctions.argsort(y_scores, reverse=True)
        y_true_sorted = [y_true[i] for i in sorted_indices]
        y_scores_sorted = [y_scores[i] for i in sorted_indices]

        tpr = [0.0]
        fpr = [0.0]
        thresholds = [y_scores_sorted[0] + 1.0] if y_scores_sorted else [1.0]

        tp = 0
        fp = 0
        n_pos = sum(1 for y in y_true if y == 1)
        n_neg = sum(1 for y in y_true if y == 0)

        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1: tp += 1
            else: fp += 1
            tpr.append(tp / n_pos if n_pos > 0 else 0)
            fpr.append(fp / n_neg if n_neg > 0 else 0)
            thresholds.append(y_scores_sorted[i])

        return fpr, tpr, thresholds

    @staticmethod
    def auc(fpr, tpr):
        """计算曲线下面积 (使用梯形法则)"""
        area = 0.0
        for i in range(len(fpr) - 1):
            area += (fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i]) / 2.0
        return area