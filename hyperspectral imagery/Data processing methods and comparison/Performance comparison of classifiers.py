import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('your_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost分类器
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost accuracy: ", accuracy_xgb)

# LightGBM分类器
lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
print("LightGBM accuracy: ", accuracy_lgbm)

# 混淆矩阵
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)

# 绘制混淆矩阵图
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(cm_xgb, annot=True, cmap='Blues', ax=ax[0])
sns.heatmap(cm_lgbm, annot=True, cmap='Blues', ax=ax[1])
ax[0].set_title("XGBoost Confusion Matrix")
ax[1].set_title("LightGBM Confusion Matrix")
plt.show()
