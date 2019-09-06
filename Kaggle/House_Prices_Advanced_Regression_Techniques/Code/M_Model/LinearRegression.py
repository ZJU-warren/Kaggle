import numpy as np


''' 
    2821357613724
    2821060482563
    2821192098387
    2821027696102
    2699186202866
    2617289275571
'''
class LinearRegression:
    theta = None

    # 注意梯度下降默认用岭回归
    def fit_gradientDescent(self, X, y, T=100000, alpha=6e-12, how='ridge', _lambda=6e-10):
        # 确定样本数目和参数数目
        n = X.shape[1]

        # 初始化theta
        self.theta = np.zeros(n)

        # 迭代T轮
        for t in range(T):
            # 预测
            hx = X.dot(self.theta)
            # 迭代
            grad = (hx - y).T.dot(X)
            if how == 'ridge':
                self.theta -= alpha * grad + _lambda * self.theta
            elif how == 'normal':
                self.theta -= alpha * grad
            print('grad:', sum(abs(grad)))

    # lasso回归
    def fit_lasso(self, X, y, T=100000, alpha=6e-12, _lambda=6e-10):
        # 确定样本数目和参数数目
        n = X.shape[1]

        # 初始化theta
        self.theta = np.ones(n)

        # 迭代T轮
        for t in range(T):
            # 预测
            hx = X.dot(self.theta)
            # 迭代
            grad = (hx - y).T.dot(X)
            sign = self.theta / abs(self.theta)
            self.theta -= alpha * grad + _lambda * sign
            # print('grad:', sum(abs(grad)))

    # 正规方程
    def fit_normal(self, X, y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # 岭回归
    def fit_ridge(self, X, y, _lambda=1):
        self.theta = np.linalg.inv(X.T.dot(X) + _lambda * np.eye(X.shape[1])).dot(X.T).dot(y)

    # 拟合
    def fit(self, X, y, how='ridge', _lambda=1):
        # 增加偏置
        X = np.append(X, np.array([[1] * X.shape[0]]).T, axis=1)

        # 根据选择拟合
        if how == 'ridge':          # Ridge回归
            self.fit_ridge(X, y, _lambda)
        elif how == 'normal':        # 正规方程
            self.fit_normal(X, y)
        elif how == 'gradientDescent':
            self.fit_gradientDescent(X, y)
        elif how == 'lasso':
            self.fit_lasso(X, y)

        # 训练集误差
        hx = X.dot(self.theta)
        err = (hx - y).dot((hx - y).T)
        print('训练集误差: %.f' % err)

    # 预测
    def predict(self, X):
        X = np.append(X, np.array([[1] * X.shape[0]]).T, axis=1)
        return X.dot(self.theta)


if __name__ == '__main__':
    n = 5
    arr = np.array([1, 9, 7])


"""

"""