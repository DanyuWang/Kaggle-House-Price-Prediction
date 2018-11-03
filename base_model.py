
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import skew
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5


RMSE = make_scorer(mean_squared_error_, greater_is_better=False)#均方根误差


def create_submission(prediction, num):
    sub_file = 'submission_' + str(num) + '.csv'
    print('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)


def data_preprocess(train, test):
    abnormal_idx = [3, 8, 12, 19, 38, 40, 46, 56, 88, 91 ,98 ,113 ,129 ,144 ,197 ,198 ,223 ,225 ,226 ,257 ,303 ,351 ,358,
                   387 ,393 ,398 ,403 ,410 ,430 ,431 ,456 ,495 ,516 ,530 ,550 ,571 ,575 ,577 ,578 , 602 ,615 ,630 ,635,
                   658 ,666 ,681 ,693 ,709 ,711 ,728 ,740 ,757 ,772 ,797 ,828 ,854 ,874 ,885 ,896 ,912 ,916 ,925 ,942,
                   944 ,951 ,968 ,970 ,978 ,995 ,1001 ,1017 ,1024 ,1032 ,1049 ,1055, 1077 ,1080 ,1099 ,1108 ,1122 ,1131,
                   1136 ,1140 ,1152 ,1182 ,1186 ,1200 ,1219 ,1220 ,1233 ,1234 ,1238 ,1245 ,1264 ,1279 ,
                   1364 ,1366 ,1413 ,1428 ,1435 ,1449 ,1450]
    train.drop(train.index[abnormal_idx], inplace=True)
    train.drop(train[(train['GrLivArea'] > 4000)].index)
    train.drop(train[(train['TotalBsmtSF'] > 6000)].index)
    train.drop(train[(train['1stFlrSF'] > 4000)].index)

    all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))

    to_delete = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    all_data = all_data.drop(to_delete, axis=1)


    train["SalePrice"] = np.log1p(train["SalePrice"]) # log transform skewed numeric features

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])#数值特征正态化

    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())

    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    return X_train, X_test, y


def model_random_forest(Xtrain, Xtest, ytrain):
    X_train = Xtrain
    y_train = ytrain
    rfr = RandomForestRegressor(n_jobs=1, random_state=0, max_depth=11)#选择n_jobs参量，结果最好

    param_grid = {'n_estimators': [300, 500], 'max_features': [10, 12, 14, 18]}
    model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)

    print('Random forest regression:')
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred


def model_gradient_boosting_tree(Xtrain, Xtest, ytrain):
    X_train = Xtrain
    y_train = ytrain
    gbr = GradientBoostingRegressor( random_state=0, max_features=10,
            learning_rate=0.05, subsample=0.8)
    param_grid = {'max_depth': [6,7,8],'n_estimators': [500,800,1000]}
    model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=20, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Gradient boosted tree regression:')
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred


def model_xgb_regression(Xtrain, Xtest, ytrain):
    X_train = Xtrain
    y_train = ytrain

    xgbreg = xgb.XGBRegressor( n_estimators=500, max_depth=7, eta = 0.05,
                              learning_rate=0.05, subsample=0.8, colsample_bytree=0.75)
    param_grid = {'seed': [0,500,1000]}
    model = GridSearchCV(estimator=xgbreg, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Extreme Gradient Boosting regression:')
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred


def model_extra_trees_regression(Xtrain, Xtest, ytrain):
    X_train = Xtrain
    y_train = ytrain

    etr = ExtraTreesRegressor(n_jobs=1, random_state=0,
            n_estimators=500, max_features=20)
    param_grid = {}
    model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Extra trees regression:')
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred

def model_ridge(Xtrain, Xtest, ytrain):
    X_train = Xtrain
    y_train = ytrain

    ridge = linear_model.Ridge()
    param_grid = {'alpha': np.logspace(-3, 2, 50)}
    model = GridSearchCV(estimator=ridge, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Ridge Regression:')
    print('Best CV score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred

def mode_ensemble_rfr_ridge(Xtrain, Xtest, ytrain):
    X_train = Xtrain
    y_train = ytrain
    X_test = Xtest

    ridge =  linear_model.Ridge(alpha=2.329952)
    rf = RandomForestRegressor(n_estimators=500, max_features=.3)
    ridge.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    y_ridge = np.expm1(ridge.predict(X_test))
    y_rf = np.expm1(rf.predict(X_test))

    y_pred = (y_ridge+y_rf)/2
    return y_pred



train = pd.read_csv("train.csv")  # read train data
test = pd.read_cPsv("test.csv")  # read test data
Xtrain, Xtest, ytrain = data_preprocess(train, test)

result=[]
test_predict1 = model_random_forest(Xtrain, Xtest, ytrain)
result.append(np.exp(test_predict1))
test_predict2 = model_gradient_boosting_tree(Xtrain, Xtest, ytrain)
result.append(np.exp(test_predict2))
test_predict3 = model_xgb_regression(Xtrain, Xtest, ytrain)
result.append(np.exp(test_predict3))
test_predict4 = model_extra_trees_regression(Xtrain, Xtest, ytrain)
result.append(np.exp(test_predict4))
test_predict5 = model_ridge(Xtrain, Xtest, ytrain)
result.append(np.exp(test_predict5))
test_predict6 = mode_ensemble_rfr_ridge(Xtrain, Xtest, ytrain)
result.append(test_predict6)
for i in range(0,result.__len__()):
    create_submission(result[i], i)


