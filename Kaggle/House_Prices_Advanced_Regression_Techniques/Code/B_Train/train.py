from Tools import *
import DataLinkSet as DLSet
from sklearn import linear_model
import pickle
from M_Model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


# 拟合模型
def TrainModel(df, model, storeLink):
    train_X = df.drop(columns=['Id', 'SalePrice']).values
    train_y = df['SalePrice'].values

    # 拟合数据
    model.fit(train_X, train_y)
    # model.fit(train_X, train_y, 'gradientDescent')
    # model.fit(train_X, train_y, 'lasso')
    # model.fit(train_X, train_y, 'ridge', _lambda=2)

    # 存储模型
    pickle.dump(model, open(storeLink, 'wb'))


# 预测结果
def Predict(df, modelLink, storeLink):
    model = pickle.load(open(modelLink, 'rb'))
    test_X = df.drop(columns=['Id']).values

    # 进行预测
    test_y = model.predict(test_X)

    # 转换格式
    pred = pd.DataFrame(test_y, columns=['SalePrice'])
    res = pd.concat([df['Id'], pred], axis=1)
    res = res.sort_values(['Id'], ascending=True)

    # 存储
    res.to_csv(storeLink, index=False)


# 主函数
def Main():
    modelStr = ('GBDT', 'std')
    dfTrain = LoadData(DLSet.new_train_link).fillna(0)
    dfTest = LoadData(DLSet.new_test_link).fillna(0)
    
    # 声明模型
    # model = linear_model.LinearRegression()
    # model = LinearRegression.LinearRegression()
    model = GradientBoostingRegressor()
    # 训练模型
    print('------------- 开始训练 -------------')
    TrainModel(dfTrain, model, DLSet.model_link % modelStr)
    # 预测结果并存储
    print('------------- 开始预测 -------------')
    Predict(dfTest, DLSet.model_link % modelStr, DLSet.res_link % modelStr)


def Run():
    Main()


if __name__ == '__main__':
    Run()

