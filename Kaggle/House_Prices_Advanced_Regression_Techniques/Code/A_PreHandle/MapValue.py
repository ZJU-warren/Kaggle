from Tools import *
import DataLinkSet as DLSet


# 混合训练集和测试集数据
def GetMix(df1, df2):
    dfTemp = df1.drop(columns=['SalePrice'])
    df = pd.concat([df1, df2], sort=False)
    return df


# 映射单个关键字
def Map(df, mapWord, storeLink):
    df = df[[mapWord]].drop_duplicates([mapWord], keep='first')
    df['flag'] = 1
    df[mapWord+'ID'] = df.groupby(['flag']).cumcount() + 1
    df.to_csv(storeLink % mapWord, index=False, columns=[mapWord+'ID', mapWord])


# 处理整张表所有关键字的映射
def HandleMap(df, mapLink):
    # 替换关键字
    keySet = df.columns.tolist()
    typeSet = df.dtypes.tolist()

    # 被映射的ID存放在mapSet中
    mapSet = []
    width = len(keySet)
    # count = 0
    for i in range(width):
        if typeSet[i] == 'object':
            # count += 1
            # print("%dth valueMap: %s" % (count, keySet[i]))
            mapSet.append(keySet[i])
            Map(df, keySet[i], mapLink)
    return mapSet


def Exg(df, mapSet, mapLink, storeLink):
    print(df.shape)
    for mapKey in mapSet:
        dfMap = LoadData(mapLink % mapKey)
        df = pd.merge(df, dfMap, on=[mapKey])
        df = df.drop(columns=[mapKey])
    print(df.shape)
    df.to_csv(storeLink, index=False)


# 主函数
def Main():
    dfTrain = LoadData(DLSet.train_link)
    dfTest = LoadData(DLSet.test_link)

    # 混合数据集合
    df = GetMix(dfTrain, dfTest)
    # 映射
    mapSet = HandleMap(df, DLSet.valueMap_link)
    # 替换
    Exg(dfTrain, mapSet, DLSet.valueMap_link, DLSet.new_train_link)
    Exg(dfTest, mapSet, DLSet.valueMap_link, DLSet.new_test_link)

def Run():
    Main()


if __name__ == '__main__':
    Run()
