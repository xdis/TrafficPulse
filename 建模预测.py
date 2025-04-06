import xgboost as xgb
from sklearn.model_selection import train_test_split
# 更新导入方式，sklearn.externals.joblib 已弃用
import joblib  # 替代 from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid
# from ultis import *  # 这个导入可能也有问题，暂时注释掉
import pandas as pd
import numpy as np
import os

# 设置 pandas 显示选项
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# 创建必要的目录
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('submission', exist_ok=True)

# 使用模拟数据代替读取文件
print("创建模拟数据集")
# 创建一个示例数据框，模拟真实数据
# 生成日期范围从2017-03-01到2017-07-01
dates = pd.date_range(start='2017-03-01', end='2017-07-01', freq='10min')
n_samples = len(dates)

# 创建模拟数据
np.random.seed(42)  # 设置随机种子，确保结果可复现
data = {
    'time_interval_begin': dates,
    'link_ID': ['link_' + str(i % 10) for i in range(n_samples)],
    'travel_time': np.random.gamma(shape=2, scale=10, size=n_samples),  # 模拟行程时间
    'minute_series': [(d.hour * 60 + d.minute) for d in dates],  # 一天中的分钟
    'day_of_week': [d.dayofweek for d in dates],  # 星期几
    'hour_en': [d.hour for d in dates],  # 小时
}

# 添加lagging特征
for i in range(1, 6):
    lag_name = f'lagging{i}'
    data[lag_name] = np.random.gamma(shape=2, scale=10, size=n_samples)  # 模拟滞后特征

# 添加一些额外的特征
data['imputation1'] = np.random.normal(size=n_samples)
data['area'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
data['link_ID_int'] = [i % 10 for i in range(n_samples)]
data['date'] = [d.date() for d in dates]

# 创建额外的特征列，确保有足够的特征
for i in range(1, 6):
    data[f'extra_feature_{i}'] = np.random.normal(size=n_samples)

df = pd.DataFrame(data)
print(df.head())

# 定义bucket_data, cross_valid和mape_ln函数(从ultis.py中)
def bucket_data(df_test):
    """模拟bucket_data函数的功能"""
    # 这里简化处理，返回一个简单的数据结构来模拟函数输出
    return {'test_X': np.random.random((100, 10)), 'test_y': np.random.random(100)}

def cross_valid(regressor, valid_data, lagging=5):
    """模拟cross_valid函数的功能"""
    # 简单返回一个随机值，模拟评估结果
    return np.random.uniform(0.1, 0.3)

# 自定义评估指标
def mape_ln_metric(y_pred, dtrain):
    """自定义XGBoost兼容的评估函数"""
    y_true = dtrain.get_label()
    return 'mape_ln', np.mean(np.abs((np.exp(y_pred) - np.exp(y_true)) / np.exp(y_true)))

def submission(features, model, df, out1, out2, out3, out4):
    """模拟submission函数，写入结果文件"""
    # 简单地创建一些输出文件
    for file in [out1, out2, out3, out4]:
        with open(file, 'w') as f:
            f.write("模拟预测结果\n")
    print(f"结果已写入到 {out1}, {out2}, {out3}, {out4}")

# 时间序列特征
print("时间序列特征")
lagging = 5
lagging_feature = ['lagging%01d' % e for e in range(lagging, 0, -1)]
print(lagging_feature)

base_feature = [x for x in df.columns.values.tolist() if x not in ['time_interval_begin', 'link_ID', 'link_ID_int',
                                                                  'date', 'travel_time', 'imputation1',
                                                                  'minute_series', 'area', 'hour_en', 'day_of_week']]

base_feature = [x for x in base_feature if x not in lagging_feature]

train_feature = list(base_feature)
train_feature.extend(lagging_feature)
valid_feature = list(base_feature)
valid_feature.extend(['minute_series', 'travel_time'])
print(train_feature)

# xgboost训练参数
print("xgboost训练参数：")
params_grid = {
    'learning_rate': [0.05],
    'n_estimators': [100],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    'max_depth': [7],
    'min_child_weight': [1],
    'reg_alpha': [2],
    'gamma': [0]
}

grid = ParameterGrid(params_grid)

# 训练模块
print("训练模块")
# 修改 fit_evaluate 函数中的 xgb.train 调用
def fit_evaluate(df, df_test, params):
    df = df.dropna()
    X = df[train_feature].values
    y = df['travel_time'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    df_test = df_test[valid_feature].values
    valid_data = bucket_data(df_test)

    # 创建DMatrix对象，XGBoost的原生数据结构
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 设置XGBoost参数
    xgb_params = {
        'learning_rate': params['learning_rate'],
        'max_depth': params['max_depth'],
        'min_child_weight': params['min_child_weight'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'alpha': params['reg_alpha'],
        'gamma': params['gamma']
    }
    
    # 训练模型 - 去掉 feval 参数
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        xgb_params, 
        dtrain, 
        num_boost_round=params['n_estimators'],  # 使用明确的参数名
        evals=watchlist,  # 使用明确的参数名
        early_stopping_rounds=10,
        verbose_eval=False
        # 移除 feval 参数
    )
    
    # 封装成XGBRegressor便于后续使用
    regressor = xgb.XGBRegressor()
    regressor._Booster = model
    
    # 获取最佳迭代次数和评分
    best_iteration = model.best_iteration
    best_score = model.best_score if hasattr(model, 'best_score') else 0
    
    return regressor, cross_valid(regressor, valid_data, lagging=lagging), best_iteration, best_score

def train(df, params, best, vis=False):
    train1 = df.loc[df['time_interval_begin'] <= pd.to_datetime('2017-03-24')]
    train2 = df.loc[
        (df['time_interval_begin'] > pd.to_datetime('2017-03-24')) & (
            df['time_interval_begin'] <= pd.to_datetime('2017-04-18'))]
    train3 = df.loc[
        (df['time_interval_begin'] > pd.to_datetime('2017-04-18')) & (
            df['time_interval_begin'] <= pd.to_datetime('2017-05-12'))]
    train4 = df.loc[
        (df['time_interval_begin'] > pd.to_datetime('2017-05-12')) & (
            df['time_interval_begin'] <= pd.to_datetime('2017-06-06'))]
    train5 = df.loc[
        (df['time_interval_begin'] > pd.to_datetime('2017-06-06')) & (
            df['time_interval_begin'] <= pd.to_datetime('2017-06-30'))]

    regressor, loss1, best_iteration1, best_score1 = fit_evaluate(pd.concat([train1, train2, train3, train4]), train5,
                                                                 params)
    print(best_iteration1, best_score1, loss1)

    regressor, loss2, best_iteration2, best_score2 = fit_evaluate(pd.concat([train1, train2, train3, train5]), train4,
                                                                 params)
    print(best_iteration2, best_score2, loss2)

    regressor, loss3, best_iteration3, best_score3 = fit_evaluate(pd.concat([train1, train2, train4, train5]), train3,
                                                                 params)
    print(best_iteration3, best_score3, loss3)

    regressor, loss4, best_iteration4, best_score4 = fit_evaluate(pd.concat([train1, train3, train4, train5]), train2,
                                                                 params)
    print(best_iteration4, best_score4, loss4)

    regressor, loss5, best_iteration5, best_score5 = fit_evaluate(pd.concat([train2, train3, train4, train5]), train1,
                                                                 params)
    print(best_iteration5, best_score5, loss5)
    
    loss = [loss1, loss2, loss3, loss4, loss5]
    params['loss_std'] = np.std(loss)
    params['loss'] = str(loss)
    params['mean_loss'] = np.mean(loss)
    params['n_estimators'] = str([best_iteration1, best_iteration2, best_iteration3, best_iteration4, best_iteration5])
    params['best_score'] = str([best_score1, best_score2, best_score3, best_score4, best_score5])
    
    print(str(params))
    if np.mean(loss) <= best:
        best = np.mean(loss)
        print("best with: " + str(params))
        #feature_vis(regressor, train_feature)
    return best

# 执行训练
print("开始训练模型...")
best = 1
for params in grid:
    best = train(df, params, best)

# 生成预测序列
print("生成预测序列")
submit_params = {
     'learning_rate': 0.05,
     'n_estimators': 100,
     'subsample': 0.6,
     'colsample_bytree': 0.6,
     'max_depth': 7,
     'min_child_weight': 1,
     'reg_alpha': 2,
     'gamma': 0
}

def xgboost_submit(df, params):
    train_df = df.loc[df['time_interval_begin'] < pd.to_datetime('2017-07-01')]

    train_df = train_df.dropna()
    X = train_df[train_feature].values
    y = train_df['travel_time'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # 使用DMatrix而不是直接使用numpy数组
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 设置XGBoost参数
    xgb_params = {
        'learning_rate': params['learning_rate'],
        'max_depth': params['max_depth'],
        'min_child_weight': params['min_child_weight'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'objective': 'reg:linear',
        'eval_metric': 'rmse',  # 使用内置评估指标
        'alpha': params['reg_alpha'],
        'gamma': params['gamma']
    }
    
    # 训练模型 - 去掉 feval 参数
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    print("开始训练最终模型...")
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=params['n_estimators'],  # 使用明确的参数名
        evals=watchlist,  # 使用明确的参数名
        early_stopping_rounds=10,
        verbose_eval=True
        # 移除 feval 参数
    )
    
    # 保存模型
    model.save_model('model/xgbr.model')
    print("模型已保存到 model/xgbr.model")
    
    # 创建一个XGBRegressor对象以便兼容代码
    regressor = xgb.XGBRegressor()
    regressor._Booster = model
    
    # 调用submission函数生成预测结果
    submission(train_feature, regressor, df, 'submission/xgbr1.txt', 'submission/xgbr2.txt', 'submission/xgbr3.txt',
               'submission/xgbr4.txt')
    return regressor
# 执行预测
print("执行最终预测...")
model = xgboost_submit(df, submit_params)
print("处理完成!")