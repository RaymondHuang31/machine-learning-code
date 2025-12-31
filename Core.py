import numpy as np
import pandas as pd
import os
import random
import json
from datetime import datetime

from Metrics import Metrics
from CrossValidation import CrossValidation
from FTRLProximal import FTRLProximal
from DecisionTreeNode import GBTClassifier
from SimpleAutoencoder import SimpleAutoencoder
from LogisticRegression import LogisticRegression
from ResultSaver import ResultSaver, Visualizer


class BoschSolution:

    def __init__(self, random_state=42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join('../RF/bosch_output/', self.timestamp)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        self.evaluator = Metrics()
        self.saver = ResultSaver(output_dir=self.exp_dir)
        self.visualizer = Visualizer(output_dir=self.exp_dir)

    def log_message(self, message):
        print(message)
        with open(os.path.join(self.exp_dir, "process.log"), "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {message}\n")

    def save_params(self, params_dict):
        with open(os.path.join(self.exp_dir, "hyperparameters.json"), "w") as f:
            json.dump(params_dict, f, indent=4)

    def prepare_features(self, df):
        self.log_message("执行特征工程...")
        df = df.copy()
        df['nan_count'] = df.isnull().sum(axis=1)
        vals = df.values
        with np.errstate(invalid='ignore'):
            df['mean'] = np.nanmean(vals, axis=1)
            df['std'] = np.nanstd(vals, axis=1)
        return df.fillna(0).values

    def load_and_prepare_all_features(self, data_dir, sample_size=5000):
        self.log_message(f"加载数据，采样大小: {sample_size}")
        num_path = os.path.join(data_dir, 'train_numeric.csv')
        df_chunk = pd.read_csv(num_path, nrows=200000)

        pos_df = df_chunk[df_chunk['Response'] == 1]
        neg_df = df_chunk[df_chunk['Response'] == 0]

        n_pos = min(len(pos_df), sample_size // 2)

        df_balanced = pd.concat([
            pos_df.sample(n=n_pos, random_state=self.random_state),
            neg_df.sample(n=n_pos, random_state=self.random_state)
        ]).sample(frac=1, random_state=self.random_state)

        y = df_balanced['Response'].values
        X_raw = df_balanced.drop(['Id', 'Response'], axis=1)

        raw_feature_names = list(X_raw.columns)
        X = self.prepare_features(X_raw)

        feature_names = raw_feature_names + ['NaN_Count', 'Row_Mean', 'Row_Std']

        return X, y, feature_names

    def run_complete_pipeline(self, data_dir='../RF'):
        params = {
            "random_state": self.random_state,
            "sample_size": 5000,
            "nrows": 200000,
            "gbt_ae_epochs": 50,
            "ae_encoding_dim": 50,
            "gbt_n_estimators": 500,
            "gbt_n_max_depth": 10,
            "gbt_FTRL_epochs": 500,
            "ftrl_alpha": 0.1,
            "blender_lr": 0.1
        }
        self.save_params(params)
        self.log_message(f"实验启动 - ID: {self.timestamp}")

        X, y, base_feature_names = self.load_and_prepare_all_features(data_dir, sample_size=params["sample_size"])
        X, y = np.array(X), np.array(y)

        ae_feature_names = [f'AE_Feature_{i}' for i in range(params["ae_encoding_dim"])]
        full_feature_names = base_feature_names + ae_feature_names

        models_metrics = {
            'FTRL': {'mcc': [], 'auc': [], 'precision': [], 'recall': []},
            'GBT': {'mcc': [], 'auc': [], 'precision': [], 'recall': []},
            'Stacking': {'mcc': [], 'auc': [], 'precision': [], 'recall': []}
        }

        global_feature_importance = np.zeros(len(full_feature_names))
        training_logs = {'ae_loss': [], 'ftrl_loss': [], 'gbt_loss': []}

        mean_fpr = np.linspace(0, 1, 100)
        stacking_tprs = []
        oof_y_true = []
        oof_y_pred = []

        def evaluate_performance(y_true, y_prob):
            fpr, tpr, _ = self.evaluator.roc_curve(y_true, y_prob)
            auc_val = self.evaluator.auc(fpr, tpr)
            best_mcc, best_th = -1, 0.5

            for th in np.linspace(0.01, 0.9, 50):
                pred = (y_prob >= th).astype(int)
                mcc = self.evaluator.matthews_corrcoef(y_true, pred)
                if mcc > best_mcc: best_mcc, best_th = mcc, th


            final_preds = (y_prob >= best_th).astype(int)
            tp = np.sum((final_preds == 1) & (y_true == 1))
            fp = np.sum((final_preds == 1) & (y_true == 0))
            fn = np.sum((final_preds == 0) & (y_true == 1))
            pre = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            return best_mcc, auc_val, pre, rec

        self.log_message("启动 5-Fold 交叉验证...")

        fold = 0
        for train_idx, val_idx in CrossValidation.stratified_kfold(y, n_splits=5, shuffle=True,
                                                                   random_state=self.random_state):
            fold += 1
            print(f"Processing Fold {fold}/5...", end='\r')

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            ae = SimpleAutoencoder(input_dim=X_train.shape[1], encoding_dim=params["ae_encoding_dim"])
            X_train_neg = X_train[y_train == 0]

            ae_hist = ae.fit(X_train_neg.tolist(), epochs=params["gbt_ae_epochs"])

            X_train_ae = np.array(ae.encode(X_train.tolist()))
            X_val_ae = np.array(ae.encode(X_val.tolist()))
            X_train_comb = np.hstack([X_train, X_train_ae])
            X_val_comb = np.hstack([X_val, X_val_ae])

            gbt = GBTClassifier(n_estimators=params["gbt_n_estimators"], max_depth=params["gbt_n_max_depth"])
            # gbt.fit(X_train_comb, y_train)
            gbt_hist = gbt.fit(X_train_comb, y_train)
            p_gbt_val = gbt.predict_proba(X_val_comb)[:, 1]
            p_gbt_train = gbt.predict_proba(X_train_comb)[:, 1]

            if hasattr(gbt, 'feature_importances_'):
                for idx, score in gbt.feature_importances_.items():
                    if idx < len(global_feature_importance):
                        global_feature_importance[idx] += score

            mcc, auc, pre, rec = evaluate_performance(y_val, p_gbt_val)
            models_metrics['GBT']['mcc'].append(mcc)
            models_metrics['GBT']['auc'].append(auc)
            models_metrics['GBT']['precision'].append(pre)
            models_metrics['GBT']['recall'].append(rec)


            ftrl = FTRLProximal(dim=X_train_comb.shape[1], alpha=params["ftrl_alpha"])

            ftrl_hist = ftrl.fit(X_train_comb.tolist(), y_train.tolist(), epochs=params["gbt_FTRL_epochs"])
            p_ftrl_val = np.array(ftrl.predict_proba(X_val_comb.tolist()))[:, 1]
            p_ftrl_train = np.array(ftrl.predict_proba(X_train_comb.tolist()))[:, 1]

            mcc, auc, pre, rec = evaluate_performance(y_val, p_ftrl_val)
            models_metrics['FTRL']['mcc'].append(mcc)
            models_metrics['FTRL']['auc'].append(auc)
            models_metrics['FTRL']['precision'].append(pre)
            models_metrics['FTRL']['recall'].append(rec)


            meta_train = np.column_stack([p_gbt_train, p_ftrl_train])
            meta_val = np.column_stack([p_gbt_val, p_ftrl_val])
            blender = LogisticRegression(learning_rate=params["blender_lr"])
            blender.fit(meta_train, y_train)
            final_probs = blender.predict_proba(meta_val)[:, 1]

            mcc, auc, pre, rec = evaluate_performance(y_val, final_probs)
            models_metrics['Stacking']['mcc'].append(mcc)
            models_metrics['Stacking']['auc'].append(auc)
            models_metrics['Stacking']['precision'].append(pre)
            models_metrics['Stacking']['recall'].append(rec)

            if fold == 1:
                training_logs['ae_loss'] = ae_hist
                training_logs['ftrl_loss'] = ftrl_hist
                training_logs['gbt_loss'] = gbt_hist


            oof_y_true.extend(y_val)
            oof_y_pred.extend(final_probs)


            fpr, tpr, _ = self.evaluator.roc_curve(y_val, final_probs)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            stacking_tprs.append(interp_tpr)

        self.log_message("\n" + "=" * 60)

        # print(f"{'模型架构':<25} | {'平均 MCC':<10} | {'平均 AUC':<10}")
        # print("-" * 55)
        # for m in ['FTRL', 'GBT', 'Stacking']:
        #     print(f"{m:<25} | {np.mean(models_metrics[m]['mcc']):.3f}      | {np.mean(models_metrics[m]['auc']):.3f}")

        print(f"{'模型架构':<25} | {'平均 MCC':<10} | {'平均 AUC':<10} | {'Precision':<10} | {'Recall':<10}")
        print("-" * 75)
        for model_name in ['FTRL', 'GBT', 'Stacking']:
            metrics = models_metrics[model_name]
            print(
                f"{model_name:<25} | {np.mean(metrics['mcc']):.3f}      | {np.mean(metrics['auc']):.3f}      | {np.mean(metrics['precision']):.3f}      | {np.mean(metrics['recall']):.3f}")


        self.log_message("正在生成可视化图表...")

        self.visualizer.plot_convergence(
            training_logs['ae_loss'],
            training_logs['ftrl_loss'],
            training_logs['gbt_loss'],
            filename='convergence_analysis.png'
        )

        global_feature_importance /= 5.0
        self.visualizer.plot_feature_importance(
            global_feature_importance,
            full_feature_names,
            top_n=15,
            filename='feature_importance.png'
        )

        self.visualizer.plot_model_comparison(
            models_metrics,
            metric_name='mcc',
            filename='model_comparison.png'
        )

        mean_tpr = np.mean(stacking_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = self.evaluator.auc(mean_fpr, mean_tpr)
        self.visualizer.plot_roc_curve(mean_fpr, mean_tpr, mean_auc, title='Stacking 5-Fold Avg ROC',
                                       filename='avg_roc.png')

        best_mcc, best_th = -1, 0.5
        for th in np.linspace(0.01, 0.9, 100):
            mcc = self.evaluator.matthews_corrcoef(oof_y_true, (np.array(oof_y_pred) >= th).astype(int))
            if mcc > best_mcc: best_mcc, best_th = mcc, th

        cm = self.evaluator.confusion_matrix(oof_y_true, (np.array(oof_y_pred) >= best_th).astype(int))
        self.visualizer.plot_confusion_matrix(
            cm,
            title=f'Confusion Matrix (Th={best_th:.2f}, MCC={best_mcc:.3f})',
            filename='stacking_confusion_matrix.png'
        )

        self.log_message(f"所有图表已生成至: {self.exp_dir}")


if __name__ == "__main__":
    data_dir = '../RF'
    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试访问路径: {os.path.abspath(data_dir)}")

    if not os.path.exists(data_dir):
        print(f"错误: 目录 {data_dir} 不存在！")

        os.makedirs(data_dir, exist_ok=True)
        print(f"已创建目录: {data_dir}")

    train_file = os.path.join(data_dir, 'train_numeric.csv')
    if not os.path.exists(train_file):
        print(f"错误: 文件 {train_file} 不存在！")
        print("请确保 train_numeric.csv 文件在正确的目录中")
    else:
        print(f"找到文件: {train_file}")
        if os.access(train_file, os.R_OK):
            print("文件可读")
        else:
            print("文件不可读，请检查权限")

    sol = BoschSolution()
    sol.run_complete_pipeline()
