import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("C:/Users/Administrator/Desktop/leef/数据处理/da-all.csv")

# 分离特征和标签
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 数据标准化
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)

# 逐步回归筛选特征
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)
selected_features_rfe = X.columns[selector.support_]

# Lasso回归筛选特征
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)
selected_features_lasso = X.columns[lasso.coef_ != 0]

# 随机森林回归筛选特征
rf = RandomForestRegressor(n_estimators=100, random_state=1)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
selected_features_rf = X.columns[importances > 0.01]

# 逻辑回归筛选特征
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)
selected_features_logreg = X.columns[logreg.coef_[0] != 0]

# 输出每个方法筛选出的特征
print('逐步回归筛选出的特征：', selected_features_rfe)
print('Lasso回归筛选出的特征：', selected_features_lasso)
print('随机森林回归筛选出的特征：', selected_features_rf)
print('逻辑回归筛选出的特征：', selected_features_logreg)

# 使用svm分类器进行训练和测试，并输出分类结果
svm_clf = svm.SVC(kernel='linear', C=1, random_state=1)
svm_clf.fit(X_train[:, selector.support_], y_train)
svm_score_rfe = svm_clf.score(X_test[:, selector.support_], y_test)

svm_clf.fit(X_train[:, lasso.coef_ != 0], y_train)
svm_score_lasso = svm_clf.score(X_test[:, lasso.coef_ != 0], y_test)

svm_clf.fit(X_train[:, importances > 0.01], y_train)
svm_score_rf = svm_clf.score(X_test[:, importances > 0.01], y_test)

svm_clf.fit(X_train[:, logreg.coef_[0] != 0], y_train)
svm_score_logreg = svm_clf.score(X_test[:, logreg.coef_[0] != 0], y_test)

# 输出分类结果
print('逐步回归+svm分类器的准确率：', svm_score_rfe)
print('Lasso回归+svm分类器的准确：', svm_score_lasso)
print('随机森林+svm分类器准确率：', svm_score_rf)
print('逻辑回归+svm分类器准确率：', svm_score_logreg)