from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn import svm
# 加载数据集
# ...

df = pd.read_csv("C:/Users/Administrator/Desktop/leef/数据处理/da.csv")

# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 定义模型
xgb_model = xgb.XGBClassifier()
lgbm_model = lgb.LGBMClassifier()
svm_model = svm.SVC()

# 训练模型
xgb_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# 预测
xgb_preds = xgb_model.predict(X_test)
lgbm_preds = lgbm_model.predict(X_test)
svm_preds = svm_model.predict(X_test)

# 评估模型
xgb_cm = confusion_matrix(y_test, xgb_preds)
lgbm_cm = confusion_matrix(y_test, lgbm_preds)
svm_cm = confusion_matrix(y_test, svm_preds)

# 绘制混淆矩阵
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

sns.heatmap(xgb_cm, annot=True, cmap='Blues', ax=ax1)
ax1.set_title('XGBoost Confusion Matrix')

sns.heatmap(lgbm_cm, annot=True, cmap='Blues', ax=ax2)
ax2.set_title('LightGBM Confusion Matrix')

sns.heatmap(svm_cm, annot=True, cmap='Blues', ax=ax3)
ax3.set_title('SVM Confusion Matrix')

plt.show()
