import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
# 读取数据集
data = pd.read_csv('C:/Users/Administrator/Desktop/leef/数据处理/da-all.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lasso回归
lasso = Lasso(random_state=42)
alphas = np.logspace(-5, 2, 100)
tuned_parameters = [{'alpha': alphas}]
n_folds = 5
clf_lasso = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf_lasso.fit(X_train, y_train)

# 输出Lasso回归的最佳参数
print("Best parameters for Lasso: {}".format(clf_lasso.best_params_))

# 保存Lasso回归的特征权重
lasso_coef = pd.DataFrame(clf_lasso.best_estimator_.coef_, columns=['lasso_coef'])
lasso_coef.to_csv('lasso_coef.csv', index=False)

# 岭回归
ridge = Ridge(random_state=42)
alphas = np.logspace(-5, 2, 100)
tuned_parameters = [{'alpha': alphas}]
n_folds = 5
clf_ridge = GridSearchCV(ridge, tuned_parameters, cv=n_folds, refit=False)
clf_ridge.fit(X_train, y_train)

# 输出岭回归的最佳参数
print("Best parameters for Ridge: {}".format(clf_ridge.best_params_))

# 保存岭回归的特征权重
ridge_coef = pd.DataFrame(clf_ridge.best_estimator_.coef_, columns=['ridge_coef'])
ridge_coef.to_csv('ridge_coef.csv', index=False)
