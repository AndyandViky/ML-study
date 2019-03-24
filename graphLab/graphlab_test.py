# -*- coding: utf-8 -*-

# 机器学习 入门（一） 使用 graphlab

# 命令行下载 graphlab
# pip install --upgrade --no-cache-dir https://get.graphlab.com/GraphLab-Create/2.1/1026530721@qq.com/6902-2DA0-F12D-EA00-AB4F-D610-469A-B3CC/GraphLab-Create-License.tar.gz

# import graphlab
#
# sf = graphlab.SFrame('people-example.csv')

# sframe 基础

# 1. 输入 sf 可以看到读取进来的数据集

# 2. sf["country"] 可以看到指定的行或者列的数据

# 3. 简单操作数据集(整列赋值)

    # 1. 赋值 sf["country"] = sf["country"] + "1"
    # 2. 直接加减 sf["age"] + 1
    # 3. 赋值函数 ： sf["age"].max() 最大值  sf["age"].mean() 平均值 。。。

# 4. graphlab.canvase 图表显示 sf.show() 执行将会打开一个新的页面

    # 设置不打开新的页面，直接在 ipython notebook中打开 graphlab.canvase.set_target("ipynb") // ipynb 就是ipython notebook
    # 这样就可以直接在 ipynb 中直接显示图表信息

# 5. sf.apply(def) 函数， 此函数传入一个函数作为参数， 执行相应的操作

    # def transform_city(city):
    #   return city+"1"
    # 例如：sf["city"].apply(transform_city)
    # 可以发现执行完成后 city 都加上了一个 "1“


# -------------------------------------------
# 机器学习 入门（二） 回归线性

# 1. 直线拟合（f = w0y + w1）,简单的线性回归， 在很多数据中选取一条`残差平方和`最小的线作为模型函数
# 残差平方和 rss = 每个数据点到线的投影的距离的平方和

# 2. 二次函数拟合(f = w0 + w1x + w2x^2)，与直线拟合计算方法一样

# 3. 过度拟合， 残差平方和几乎为0。 （不符合常理的拟合）


# 如何选择模型阶数/复杂度

# 1. 模拟预测 （在足够的观察数据情况下）
    # 训练/测试分离  进行训练的数据为 `训练集`  进行测试的数据为 `测试集`

        # 1.在现有数据中移除一部分数据
        # 2.进行拟合模型
        # 3.最后预测剩下的数据，判断我们拟合的模型是否符合未知数据的预测

    # 训练误差 （称为训练集上的残差平方和） 随着复杂度的增大而减小

    # 测试误差 （成为测试集上的残差平方和） 随着复杂度的增大减小，但是在一个特定的点过后误差将会增大

# 2. 增加高阶特征参数（不再局限于线性）


# --------------------------------------------
# 机器学习 入门（三） 线性模型实践

# import graphlab
#
# sales = graphlab.SFrame("datasets/home_data.gl")
#
# # 分割训练集 和 测试集
# train_data, test_data = sales.random_split(.8, seed=0)  # 第一个参数表示分配比例， 设置 seed =0 表示当再次执行这条语句的时候，所得到的集合是一样的
#
# # 创建模型
# sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'])

# 第一个参数为 训练数据
# 第二个参数为 输出目标
# 第三个参数为 选取的相关特征参数


# --------------------------------------------
# 机器学习 入门（四） 聚类

import graphlab

datas = graphlab.SFrame('datasets/people_wiki.gl')

datas.head()
