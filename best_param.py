from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from scipy.stats import skew
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def data_preprocess(train, test):
    outlier_idx = [3, 8, 12, 19, 38, 40, 46, 56, 88, 91 ,98 ,113 ,129 ,144 ,197 ,198 ,223 ,225 ,226 ,257 ,303 ,351 ,358,
                   387 ,393 ,398 ,403 ,410 ,430 ,431 ,456 ,495 ,516 ,530 ,550 ,571 ,575 ,577 ,578 , 602 ,615 ,630 ,635,
                   658 ,666 ,681 ,693 ,709 ,711 ,728 ,740 ,757 ,772 ,797 ,828 ,854 ,874 ,885 ,896 ,912 ,916 ,925 ,942,
                   944 ,951 ,968 ,970 ,978 ,995 ,1001 ,1017 ,1024 ,1032 ,1049 ,1055, 1077 ,1080 ,1099 ,1108 ,1122 ,1131,
                   1136 ,1140 ,1152 ,1182 ,1186 ,1200 ,1219 ,1220 ,1233 ,1234 ,1238 ,1245 ,1264 ,1279 ,
                   1364 ,1366 ,1413 ,1428 ,1435 ,1449 ,1450]

    train.drop(train.index[outlier_idx], inplace=True)
    all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))

    to_delete = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

    all_data = all_data.drop(to_delete, axis=1) #合并的目的是后续的特征处理时，就一并把训练集和测试集的一并转换了。

    train["SalePrice"] = np.log1p(train["SalePrice"]) # log transform skewed numeric features

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])#之所以看分布和做转换，是因为用线性回归有个条件是因变量符合正态分布，
    # 如果不是的话，最好做个平滑处理，让其尽可能符合高斯分布，否则效果不会好。

    all_data = pd.get_dummies(all_data)#变换数据类型
    all_data = all_data.fillna(all_data.mean())#用平均值填充缺失值

    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    return X_train, X_test, y

def alpha_estimate(X_train, y_train):
    Xtrain = X_train
    ytrain = y_train

    alphas = np.logspace(-3, 2, 50)
    test_scores = []
    for alpha in alphas:
        clf = linear_model.Ridge(alpha)
        test_score = np.sqrt(-cross_val_score(clf, Xtrain, ytrain, cv=10, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    plt.plot(alphas, test_scores)
    df = pd.DataFrame(alphas, test_scores)
    print(df)
    plt.title('Alpha vs CV Error')
    plt.show()

def rft_estimate(X_train, y_train):
    Xtrain = X_train
    ytrain = y_train

    max_features = [.1, .3, .5, .7, 1, 2]
    test_scores = []
    for max_feat in max_features:
        clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
        test_score = np.sqrt(-cross_val_score(clf, Xtrain, ytrain, cv=5, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    plt.plot(max_features, test_scores)
    plt.title('Max Features vs CV Error')
    df = pd.DataFrame(max_features, test_scores)
    print(df)
    plt.show()

def gbt_estimate(X_train, y_train):
    Xtrain = X_train
    ytrain = y_train

    learning_rate = [.01, .03, .05, .07, .1, .2]
    test_scores = []
    for max_lr in  learning_rate:
        clf = GradientBoostingRegressor(n_estimators=500, max_features=max_lr)
        test_score = np.sqrt(-cross_val_score(clf, Xtrain, ytrain, cv=5, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    plt.plot(learning_rate, test_scores)
    plt.title('Max Features vs CV Error')
    df = pd.DataFrame( learning_rate, test_scores)
    print(df)
    plt.show()

def xgb_estimate(X_train, y_train):
    Xtrain = X_train
    ytrain = y_train

    depth_rate = range(5,11)
    test_scores = []
    for max_lr in  depth_rate:
        model = xgb.XGBRegressor(n_estimators=500, max_features=max_lr)
        test_score = np.sqrt(-cross_val_score(model, Xtrain, ytrain, cv=10, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    plt.plot(depth_rate, test_scores)
    plt.title('Depth Rate vs CV Error')
    df = pd.DataFrame( depth_rate, test_scores)
    print(df)
    plt.show()


train = pd.read_csv("train.csv")  # read train data
test = pd.read_csv("test.csv")  # read test data
Xtrain, Xtest, ytrain = data_preprocess(train, test)

rft_estimate(Xtrain, ytrain)