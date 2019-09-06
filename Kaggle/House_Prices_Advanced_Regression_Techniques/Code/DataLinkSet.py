""" -------------------------- 数据仓库地址 -------------------------- """
DataSet_link = "../../DataSet"                  # 主仓库
OrgSet_link = DataSet_link + "/OrgSet"          # 原始数据仓库
MapSet_link = DataSet_link + "/MapSet"          # 映射数据仓库
NewSet_link = DataSet_link + "/NewSet"          # 新数据仓库
ModelSet_link = DataSet_link + "/ModelSet"      # 模型数据仓库
ResSet_link = DataSet_link + "/ResSet"          # 存储各种模型预测结果
""" -------------------------- 原始数据仓库 -------------------------- """
train_link = OrgSet_link + "/train.csv"         # 训练集
test_link = OrgSet_link + "/test.csv"           # 测试集

""" -------------------------- 映射数据仓库 -------------------------- """
valueMap_link = MapSet_link + '/valueMap_%s'    # 关键字映射

""" -------------------------- 新数据仓库 -------------------------- """
new_train_link = NewSet_link + "/new_train"     # 映射后的训练集
new_test_link = NewSet_link + "/new_test"       # 映射后的测试集

""" -------------------------- 模型数据仓库 -------------------------- """
model_link = ModelSet_link + "/%s_model_%s"     # std\my模型

""" -------------------------- 预测结果仓库 -------------------------- """
res_link = ResSet_link + "/result_%s_%s"           # 存储各类模型预测结果

# ------------------------------------------------------------------
from Tools import *

if __name__ == '__main__':
    df = LoadData(train_link[3:])
    print(df.head(5))
    print(df.columns.tolist())
