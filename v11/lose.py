import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
#from process import normalize_data
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn import preprocessing
import csv
import matplotlib.pyplot as plt


torch.manual_seed(1)
#*************train data*******************
train_df = pd.read_csv("./train.csv", index_col = 0)
test_df = pd.read_csv("./test.csv", index_col=0)#讀入測試數據
prices = pd.DataFrame({"price":train_df["total_price"],
"log(price + 1)":np.log1p(train_df["total_price"])})
prices.hist()
y_train = np.log1p(train_df.pop('total_price'))#將訓練數據價格列彈出（減少一列），進行加1取對數處理，同prices第二列
all_df = pd.concat((train_df, test_df), axis=0)
all_df.shape#展示一下規模
y_train.head()#展示前幾個訓練集y數據
all_df['building_material'].dtypes#查看building_material列的數據類型

print("{}".format(all_df['building_material'].dtypes))

all_df['building_material'] = all_df['building_material'].astype(str)#類型強制轉換爲字符串
all_df['building_material'].value_counts()#計數，相同的有幾次,按倒序排列
temp = pd.get_dummies(all_df['building_material'], prefix='building_material').head()#get_dummies構建詞向量表示，座標爲詞集合
all_dummy_df = pd.get_dummies(all_df)#將object數據類型的列都展開爲二值表示，因爲MSSubClass的數值沒啥意義（住宅類型，類似於一種型號編碼），也轉

all_df['building_type'] = all_df['building_type'].astype(str)#類型強制轉換爲字符串
all_df['building_type'].value_counts()#計數，相同的有幾次,按倒序排列
temp = pd.get_dummies(all_df['building_type'], prefix='building_type').head()#get_dummies構建詞向量表示，座標爲詞集合
all_dummy_df = pd.get_dummies(all_df)#將object數據類型的列都展開爲二值表示，因爲MSSubClass的數值沒啥意義（住宅類型，類似於一種型號編碼），也轉

all_df['city'] = all_df['city'].astype(str)#類型強制轉換爲字符串
all_df['city'].value_counts()#計數，相同的有幾次,按倒序排列
temp = pd.get_dummies(all_df['city'], prefix='city').head()#get_dummies構建詞向量表示，座標爲詞集合
all_dummy_df = pd.get_dummies(all_df)#將object數據類型的列都展開爲二值表示，因爲MSSubClass的數值沒啥意義（住宅類型，類似於一種型號編碼），也轉

all_df['town'] = all_df['town'].astype(str)#類型強制轉換爲字符串
all_df['town'].value_counts()#計數，相同的有幾次,按倒序排列
temp = pd.get_dummies(all_df['town'], prefix='town').head()#get_dummies構建詞向量表示，座標爲詞集合
all_dummy_df = pd.get_dummies(all_df)#將object數據類型的列都展開爲二值表示，因爲MSSubClass的數值沒啥意義（住宅類型，類似於一種型號編碼），也轉

all_df['village'] = all_df['village'].astype(str)#類型強制轉換爲字符串
all_df['village'].value_counts()#計數，相同的有幾次,按倒序排列
temp = pd.get_dummies(all_df['village'], prefix='village').head()#get_dummies構建詞向量表示，座標爲詞集合
all_dummy_df = pd.get_dummies(all_df)#將object數據類型的列都展開爲二值表示，因爲MSSubClass的數值沒啥意義（住宅類型，類似於一種型號編碼），也轉

all_df['building_use'] = all_df['building_use'].astype(str)#類型強制轉換爲字符串
all_df['building_use'].value_counts()#計數，相同的有幾次,按倒序排列
temp = pd.get_dummies(all_df['building_use'], prefix='building_use').head()#get_dummies構建詞向量表示，座標爲詞集合
all_dummy_df = pd.get_dummies(all_df)#將object數據類型的列都展開爲二值表示，因爲MSSubClass的數值沒啥意義（住宅類型，類似於一種型號編碼），也轉

all_df['parking_way'] = all_df['parking_way'].astype(str)#類型強制轉換爲字符串
all_df['parking_way'].value_counts()#計數，相同的有幾次,按倒序排列
temp = pd.get_dummies(all_df['parking_way'], prefix='parking_way').head()#get_dummies構建詞向量表示，座標爲詞集合
all_dummy_df = pd.get_dummies(all_df)#將object數據類型的列都展開爲二值表示，因爲MSSubClass的數值沒啥意義（住宅類型，類似於一種型號編碼），也轉

print("{}".format(all_dummy_df))

all_dummy_df.head()#展示前5行
all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)#計算每一列的缺失按降序排列
mean_cols = all_dummy_df.mean()#計算每一列的平均值
mean_cols.head(10)#展示平均值的前10個
all_dummy_df = all_dummy_df.fillna(mean_cols)#用平均值填補缺失
all_dummy_df.isnull().sum().sum()#看看還有沒有缺失
numeric_cols = all_df.columns[all_df.dtypes != 'object']#找出數值類型的列，這時候building_material已經處理了

numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()#loc——通過行標籤索引行數據 ，求數值列的均值
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()#求數值列的標準差
#all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std#把dummy數據的數值列標準化，即均值除以標準差
min_max_scaler = preprocessing.MinMaxScaler()
all_dummy_df.loc[:, numeric_cols] = min_max_scaler.fit_transform((all_dummy_df.loc[:, numeric_cols] - numeric_col_means))
dummy_train_df = all_dummy_df.loc[train_df.index]#取出dummy的訓練數據
dummy_test_df = all_dummy_df.loc[test_df.index]#取出dummy的測試數據
dummy_train_df.shape, dummy_test_df.shape#看看他們的形狀


#***********************Ridge Regression************************************************

from sklearn.linear_model import Ridge#從sklearn中導入嶺迴歸的包
from sklearn.model_selection import cross_val_score#從sklearn中導入交叉驗證的包

X_train = dummy_train_df.values#訓練集所有數據數值化
X_test = dummy_test_df.values#測試集所有數據數值化
alphas = np.logspace(-3, 2, 50)#先等距劃分，再以10爲底指數化，就是說log完後是線性的,讓嶺迴歸的懲罰係數呈指數增長
test_scores = []#用空集初始化

for alpha in alphas:#使用交叉驗證，遍歷嶺迴歸係數，選出最優參數
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
import matplotlib.pyplot as plt#導入pyplot
#%matplotlib inline
#plt.plot(alphas, test_scores)#看圖直觀看嶺迴歸係數和均方誤差的關係，找出最低值點
#plt.title("Alpha vs CV Error")
#plt.show()
ridge_param = alphas[test_scores.index(min(test_scores))]#最佳參數約爲19.307
#***********************Ridge Regression************************************************


ridge = Ridge(alpha=ridge_param)#alpha爲15的嶺迴歸，這是前面測出來的
ridge.fit(X_train, y_train)#嶺迴歸擬合
y_ridge = np.expm1(ridge.predict(X_test))#預測結果取e指數減1（log1p的逆過程）
submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_ridge})
#構造提交文檔
submission_df.head(10)
submission_df.to_csv('sub_result.csv',index=False,sep=',')#保存