import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
import warnings
import xgboost as xgb
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import math
#import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from scipy.signal import welch
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import pywt
from xgboost import XGBClassifier as XGBC
from tqdm import tqdm
from xgboost import XGBRegressor as XGBR
import datetime
from collections import defaultdict

class EnhancedXGBoost:
    def __init__(self, default_history=0.0):
        self.clf = XGBC()  # 分类模型
        self.reg = XGBR()  # 回归模型
        self.history = {}  # 历史记录 {identifier: score}
        self.default_history = default_history
        self.first_run = True
    def _get_history(self, identifiers):
        """获取历史分数并更新标识符"""
        self.current_ids = identifiers
        return np.array([self.history.get(id, self.default_history)
                         for id in identifiers]).reshape(-1, 1)

    def fit(self, X, y, identifiers, eval_set=None):
        """
        X: 特征矩阵
        y: 标签 (0/1)
        identifiers: 样本唯一标识列表
        eval_set: 可选验证集 (格式同输入)
        """
        # 获取历史分数并拼接
        # print(identifiers)
        hist = self._get_history(identifiers)
        X_enhanced = np.hstack([X, hist])



        # 训练回归模型（错误分数预测）
        # 假设回归目标为预测误差的平方
        if self.first_run:
            self.reg.fit(X_enhanced, y, eval_set=eval_set)
            self.first_run = False
            y_pred = self.reg.predict(X_enhanced)
        else:
            y_pred = self.reg.predict(X_enhanced)
        # regression_target = (y - y_pred) ** 2  # 自定义回归目标
        X['y'] = y
        # X_new = np.hstack([X, y_pred])

        # 训练分类模型（标签预测）
        self.clf.fit(X, y, eval_set=eval_set)
        y_clf_pred = self.clf.predict(X)
        regression_target = (y - y_clf_pred) ** 2  # 自定义回归目标
        self.reg.fit(X_enhanced, regression_target, eval_set=eval_set)
    def predict(self, X, identifiers):
        # 获取历史分数并拼接
        hist = self._get_history(identifiers)
        X_enhanced = np.hstack([X, hist])

        # 预测标签
        y_pred = self.clf.predict(X_enhanced)

        # 预测新错误分数
        new_scores = self.reg.predict(X_enhanced)

        # 更新历史记录
        for id, score in zip(identifiers, new_scores):
            self.history[id] = score

        return y_pred

    def get_history(self, identifier):
        return self.history.get(identifier, self.default_history)


class HistoryEnhancedClassifier:
    def __init__(self, decay_factor=0.9, max_history=5):
        """
        :param decay_factor: 历史错误衰减系数
        :param max_history: 最大历史记录长度
        """
        self.clf = XGBC()
        self.history = defaultdict(lambda: np.zeros(max_history))
        self.decay = decay_factor
        self.max_history = max_history

    def _get_history_features(self, identifiers):
        """生成动态历史特征"""
        features = []
        for id in identifiers:
            # 应用时间衰减 (最新错误权重更大)
            decayed = [self.decay ** i * err for i, err in enumerate(self.history[id])]
            features.append([
                np.mean(decayed),  # 衰减后均值
                np.max(decayed),  # 最大历史错误
                np.min(decayed),  # 最小历史错误
                len(self.history[id]) / self.max_history  # 历史密度
            ])
        return np.array(features)

    def fit(self, X, y, identifiers):
        # 生成历史增强特征
        hist_feats = self._get_history_features(identifiers)
        X_enhanced = np.hstack([X, hist_feats])

        # 训练分类器
        self.clf.fit(X_enhanced, y)

        # 记录训练集的初始错误（用于后续更新）
        self._update_history(X_enhanced, y, identifiers)

    def predict(self, X, identifiers):
        # 生成历史特征
        hist_feats = self._get_history_features(identifiers)
        X_enhanced = np.hstack([X, hist_feats])

        # 预测并更新历史
        proba = self.clf.predict_proba(X_enhanced)[:, 1]
        preds = (proba > 0.5).astype(int)
        self._update_history(X_enhanced, preds, identifiers, is_train=False)

        return preds

    def _update_history(self, X, y, identifiers, is_train=True):
        # 获取预测结果
        proba = self.clf.predict_proba(X)[:, 1]

        # 计算错误分数（基于预测置信度）
        for i, id in enumerate(identifiers):
            true_label = y[i]
            confidence = proba[i] if true_label == 1 else 1 - proba[i]
            error_score = 1 - confidence  # 错误分数计算

            # 更新历史记录（FIFO队列）
            self.history[id] = np.roll(self.history[id], -1)
            self.history[id][-1] = error_score

            # 如果是训练阶段，执行额外验证
            if is_train:
                current_pred = (proba[i] > 0.5).astype(int)
                if current_pred != true_label:
                    print(f"训练样本 {id} 预测错误，更新历史分数至 {self.history[id]}")

    def get_feature_importance(self):
        """获取增强后的特征重要性"""
        original_feats = self.clf.get_booster().feature_names[:X.shape[1]]
        history_feats = ['hist_mean', 'hist_max', 'hist_min', 'hist_density']
        return dict(zip(original_feats + history_feats,
                        self.clf.feature_importances_))
def create_data():
    np.random.seed(42)

    # 生成时间列（2010-01-01 到 2020-01-01 之间的随机时间）
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime(2020, 1, 1)
    delta_seconds = int((end_date - start_date).total_seconds())

    # 生成2万个随机时间戳
    random_seconds = np.random.randint(0, delta_seconds, size=20000)
    timestamps = [start_date + datetime.timedelta(seconds=int(s)) for s in random_seconds]

    # 创建DataFrame
    df = pd.DataFrame({
        "time": [ts.strftime("%Y/%m/%d %H:%M:%S") for ts in timestamps],
        "mb": np.random.randint(0, 3, 20000),  # 0-2
        "dev": np.random.randint(0, 4, 20000),  # 0-3
        "bank": np.random.randint(1, 5, 20000),  # 1-4
        "row": np.random.randint(0, 20001, 20000),  # 0-20000
        "mark": np.random.randint(0, 2, 20000)  # 0-1
    })

    # 保存为CSV文件
    df.to_csv("test_data.csv", index=False)

    print("测试数据已生成并保存为 test_data.csv")
    print(f"数据总条数：{len(df)}")
    print("数据示例：")
    print(df.head())

if __name__ == '__main__':
    # create_data()
    df = pd.read_csv("test_data.csv")
    print(df.head())

    model = EnhancedXGBoost(default_history=0.5)
    # model = HistoryEnhancedClassifier(max_history=5)
    X = df.iloc[:, 1:-1]
    y = df["mark"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_ids = (
        X_train[['mb', 'dev', 'bank']]
        .astype(str)  # 确保数据类型统一
        .apply(lambda row: f"mb{row['mb']}_dev{row['dev']}_bank{row['bank']}", axis=1)
        .values
    )
    test_ids = (
        X_test[['mb', 'dev', 'bank']]
        .astype(str)  # 确保数据类型统一
        .apply(lambda row: f"mb{row['mb']}_dev{row['dev']}_bank{row['bank']}", axis=1)
        .values
    )
    model.fit(X_train, y_train, train_ids)
    # print(model.get_feature_importance())
    predictions = model.predict(X_test, test_ids)
    print("测试集准确率：", np.mean(predictions == y_test))
    print("历史记录示例：")
    print(model.history)

