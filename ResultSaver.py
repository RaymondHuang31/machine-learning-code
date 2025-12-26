import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResultSaver:
    def __init__(self, output_dir='../RF/bosch_output/'):
        self.output_dir = output_dir
        # 递归创建目录
        os.makedirs(output_dir, exist_ok=True)
        self.plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

    def save_metrics(self, metrics_dict, filename='metrics.txt'):
        """保存数值指标到文本文件"""
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for k, v in metrics_dict.items():
                f.write(f"{k}: {v}\n")


class Visualizer:
    """支持自动保存的可视化类"""

    def __init__(self, output_dir='../RF/bosch_output/'):
        self.output_dir = output_dir
        self.plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

    def plot_convergence(self, ae_history, ftrl_history, gbt_history, filename='convergence_analysis.png'):
        """绘制 AE、FTRL 和 GBT 的收敛曲线 (三子图)"""
        # 修改为 1行3列
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # 子图 1: Autoencoder Loss
        if ae_history:
            ax1.plot(ae_history, label='Reconstruction MSE', color='#1f77b4', linewidth=2)
            ax1.set_title('AE Convergence (Unsupervised)', fontsize=12)
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('MSE')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()

        # 子图 2: FTRL Loss
        if ftrl_history:
            ax2.plot(ftrl_history, label='Log Loss', color='#ff7f0e', linewidth=2)
            ax2.set_title('FTRL Online Learning Curve', fontsize=12)
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Log Loss')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()

        # 子图 3: GBT Loss 【新增】
        if gbt_history:
            ax3.plot(gbt_history, label='Training Log Loss', color='#2ca02c', linewidth=2)
            ax3.set_title('GBT Boosting Process', fontsize=12)
            ax3.set_xlabel('Trees (Iterations)')
            ax3.set_ylabel('Log Loss')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()

        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_confusion_matrix(self, cm, classes=['Good', 'Defect'], title='Confusion Matrix', filename='confusion_matrix.png'):
        cm = np.array(cm)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        save_path = os.path.join(self.plot_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_roc_curve(self, fpr, tpr, auc, title='ROC Curve', filename='roc_curve.png'):
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.5)

        save_path = os.path.join(self.plot_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_feature_importance(self, importances, feature_names, top_n=10, filename='feature_importance.png'):
        """绘制特征重要性 Top-N"""
        # 排序
        sorted_idx = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in sorted_idx]
        top_scores = [importances[i] for i in sorted_idx]

        plt.figure(figsize=(10, 6))
        # 倒序以便重要性最高的在上面
        plt.barh(range(len(top_scores)), top_scores[::-1], align='center', color='skyblue')
        plt.yticks(range(len(top_scores)), top_features[::-1])
        plt.xlabel('Relative Importance Score (Gain)')
        plt.title(f'Top-{top_n} Feature Importance (GBT Analysis)')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = os.path.join(self.plot_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_model_comparison(self, metrics_dict, metric_name='mcc', filename='model_comparison.png'):
        """绘制多模型性能对比柱状图"""
        models = list(metrics_dict.keys())
        scores = [np.mean(metrics_dict[m][metric_name]) for m in models]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(models, scores, color=['#aaccff', '#99ddff', '#ffcc99'], width=0.6)

        plt.ylabel(metric_name.upper())
        plt.title(f'Ablation Study: Performance Comparison ({metric_name.upper()})')
        plt.ylim(0, max(scores) * 1.3)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()