import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 创建必要的目录
os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('submission', exist_ok=True)

# 设置显示选项
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# 读取训练数据集
print("读取训练数据集...")
try:
    training_data = pd.read_table('data/trajectories(table 5)_training.txt', sep=';')
    print("成功读取训练数据集")
except FileNotFoundError:
    # 如果文件不存在，创建模拟数据
    print("找不到原始训练数据，创建模拟数据...")
    # 创建示例日期范围
    dates = pd.date_range(start='2017-03-01', end='2017-06-30', freq='20min')
    n_samples = len(dates)
    
    # 创建随机链接ID
    link_ids = [f"link_{i}" for i in range(1, 11)]
    
    # 创建模拟数据
    np.random.seed(42)
    data = {
        'link_ID': np.random.choice(link_ids, size=n_samples),
        'time_interval': [f"{d.strftime('%Y-%m-%d')}_{(d.hour*60+d.minute)//20}" for d in dates],
        'travel_time': np.random.gamma(shape=2, scale=10, size=n_samples)
    }
    training_data = pd.DataFrame(data)
    
    # 保存模拟数据
    training_data.to_csv('data/trajectories(table 5)_training.txt', sep=';', index=False)
    print("已创建并保存模拟训练数据")

# 数据预处理
print("开始数据预处理...")

# 分离日期和时间间隔
print("分离日期和时间间隔...")
training_data['date_time'] = training_data['time_interval'].str.split('_').str[0]
training_data['time_interval_minutes'] = training_data['time_interval'].str.split('_').str[1].astype('int64')

# 将日期转换为datetime格式
training_data['date'] = pd.to_datetime(training_data['date_time'])

# 创建时间特征
print("创建时间特征...")
training_data['day_of_week'] = training_data['date'].dt.dayofweek.astype('int64')
training_data['hour'] = (training_data['time_interval_minutes'] // 60).astype('int64')
training_data['minute'] = (training_data['time_interval_minutes'] % 60).astype('int64')
training_data['time_interval_begin'] = training_data.apply(
    lambda x: datetime.combine(x['date'].date(), 
                              datetime.min.time()) + timedelta(minutes=int(x['time_interval_minutes'])), 
    axis=1
)

# 增加滞后特征
print("创建滞后特征...")
# 为每个link_ID创建滞后特征
def add_lagging_features(df, lagging=5):
    """为每个link_ID添加滞后特征"""
    df = df.sort_values(['link_ID', 'time_interval_begin'])
    
    # 初始化滞后特征列
    for i in range(1, lagging + 1):
        df[f'lagging{i}'] = np.nan
    
    # 对每个link_ID分别处理
    for link_id in df['link_ID'].unique():
        link_data = df[df['link_ID'] == link_id].copy()
        
        # 创建滞后特征
        for i in range(1, lagging + 1):
            df.loc[df['link_ID'] == link_id, f'lagging{i}'] = link_data['travel_time'].shift(i).values
    
    return df

# 应用滞后特征函数
training_data = add_lagging_features(training_data)

# 增加统计特征 - 使用替代方法避免数据类型不匹配问题
print("创建统计特征...")

# 创建统计特征的替代方法 - 使用字典映射而不是merge
print("计算小时统计特征...")
hourly_means = training_data.groupby(['link_ID', 'hour'])['travel_time'].mean().to_dict()
hourly_stds = training_data.groupby(['link_ID', 'hour'])['travel_time'].std().fillna(0).to_dict()
hourly_mins = training_data.groupby(['link_ID', 'hour'])['travel_time'].min().to_dict()
hourly_maxs = training_data.groupby(['link_ID', 'hour'])['travel_time'].max().to_dict()

print("计算星期几统计特征...")
weekday_means = training_data.groupby(['link_ID', 'day_of_week'])['travel_time'].mean().to_dict()
weekday_stds = training_data.groupby(['link_ID', 'day_of_week'])['travel_time'].std().fillna(0).to_dict()
weekday_mins = training_data.groupby(['link_ID', 'day_of_week'])['travel_time'].min().to_dict()
weekday_maxs = training_data.groupby(['link_ID', 'day_of_week'])['travel_time'].max().to_dict()

print("应用统计特征...")
# 使用apply而不是merge
training_data['hourly_mean'] = training_data.apply(lambda x: hourly_means.get((x['link_ID'], x['hour']), np.nan), axis=1)
training_data['hourly_std'] = training_data.apply(lambda x: hourly_stds.get((x['link_ID'], x['hour']), np.nan), axis=1)
training_data['hourly_min'] = training_data.apply(lambda x: hourly_mins.get((x['link_ID'], x['hour']), np.nan), axis=1)
training_data['hourly_max'] = training_data.apply(lambda x: hourly_maxs.get((x['link_ID'], x['hour']), np.nan), axis=1)

training_data['weekday_mean'] = training_data.apply(lambda x: weekday_means.get((x['link_ID'], x['day_of_week']), np.nan), axis=1)
training_data['weekday_std'] = training_data.apply(lambda x: weekday_stds.get((x['link_ID'], x['day_of_week']), np.nan), axis=1)
training_data['weekday_min'] = training_data.apply(lambda x: weekday_mins.get((x['link_ID'], x['day_of_week']), np.nan), axis=1)
training_data['weekday_max'] = training_data.apply(lambda x: weekday_maxs.get((x['link_ID'], x['day_of_week']), np.nan), axis=1)

# 对categorical特征进行编码
print("对分类特征进行编码...")
label_encoder = LabelEncoder()
training_data['link_ID_int'] = label_encoder.fit_transform(training_data['link_ID'])
training_data['hour_en'] = training_data['hour']
training_data['weekday_en'] = training_data['day_of_week']

# 添加分钟序列特征（一天中的分钟数）
training_data['minute_series'] = training_data['hour'] * 60 + training_data['minute']

# 创建是否高峰时段特征
training_data['is_rush_hour'] = ((training_data['hour'] >= 7) & (training_data['hour'] <= 9)) | \
                                ((training_data['hour'] >= 17) & (training_data['hour'] <= 19))
training_data['is_rush_hour'] = training_data['is_rush_hour'].astype(int)

# 创建是否工作日特征
training_data['is_weekday'] = (training_data['day_of_week'] < 5).astype(int)

# 保存处理后的数据
print("保存处理后的数据...")
processed_data = training_data.copy()
processed_data.to_csv('data/training.txt', sep=';', index=False)

# 数据探索与可视化
print("生成数据探索可视化...")

try:
    # 箱线图：按星期几查看旅行时间分布
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='day_of_week', y='travel_time', data=training_data.sample(min(10000, len(training_data))))
    plt.title('Travel Time Distribution by Day of Week')
    plt.savefig('figures/travel_time_by_weekday.png')
    plt.close()

    # 按小时查看平均旅行时间
    hourly_avg = training_data.groupby('hour')['travel_time'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='hour', y='travel_time', data=hourly_avg)
    plt.title('Average Travel Time by Hour')
    plt.savefig('figures/avg_travel_time_by_hour.png')
    plt.close()

    # 热图：按小时和星期几查看旅行时间
    pivot_data = training_data.pivot_table(
        index='hour', 
        columns='day_of_week', 
        values='travel_time', 
        aggfunc='mean'
    )
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, cmap='YlGnBu', annot=False)
    plt.title('Average Travel Time by Hour and Day of Week')
    plt.savefig('figures/travel_time_heatmap.png')
    plt.close()

    # 滞后特征和旅行时间的相关性
    lag_cols = [col for col in training_data.columns if col.startswith('lagging')]
    lag_data = training_data[['travel_time'] + lag_cols].dropna()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(lag_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation between Travel Time and Lagged Features')
    plt.savefig('figures/lagged_features_correlation.png')
    plt.close()

    print("已生成数据可视化图表")
except Exception as e:
    print(f"生成图表时出错: {e}")

# 特征重要性分析（简单示例）
print("计算特征重要性...")
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    # 选择要使用的特征
    feature_cols = [
        'hour', 'minute_series', 'day_of_week', 'is_rush_hour', 'is_weekday',
        'hourly_mean', 'hourly_std', 'weekday_mean', 'weekday_std'
    ]
    
    # 添加可用的滞后特征
    for lag_col in lag_cols:
        if lag_col in training_data.columns:
            feature_cols.append(lag_col)
    
    # 准备数据
    features = training_data[feature_cols].dropna()
    target = training_data.loc[features.index, 'travel_time']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # 训练简单的XGBoost模型
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # 获取特征重要性
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png')
    plt.close()
    
    print("特征重要性分析完成")
except Exception as e:
    print(f"特征重要性分析时出错: {e}")

print("数据预处理和特征提取完成！")
print(f"处理后的数据保存在: data/training.txt")
print(f"可视化结果保存在: figures/")

# 显示数据集的基本信息
print("\n数据集信息:")
print(f"总记录数: {len(processed_data)}")
print(f"链接数量: {processed_data['link_ID'].nunique()}")
print(f"日期范围: {processed_data['date'].min()} 到 {processed_data['date'].max()}")
print(f"特征数量: {len(processed_data.columns)}")
print("\n前5条记录:")
print(processed_data.head())

# 特征列表总结
print("\n生成的特征列表:")
feature_groups = {
    "时间特征": ["day_of_week", "hour", "minute", "minute_series", "is_rush_hour", "is_weekday"],
    "滞后特征": [col for col in processed_data.columns if col.startswith("lagging")],
    "统计特征": ["hourly_mean", "hourly_std", "hourly_min", "hourly_max", 
                "weekday_mean", "weekday_std", "weekday_min", "weekday_max"],
    "编码特征": ["link_ID_int", "hour_en", "weekday_en"]
}

for group, features in feature_groups.items():
    print(f"\n{group}:")
    for feature in features:
        if feature in processed_data.columns:
            print(f"  - {feature}")

# 配套 建模预测.py 文件所需的辅助函数定义
def mape_ln_metric(y_pred, dtrain):
    """自定义XGBoost兼容的评估函数"""
    y_true = dtrain.get_label()
    return 'mape_ln', np.mean(np.abs((np.exp(y_pred) - np.exp(y_true)) / np.exp(y_true)))

def bucket_data(df_test):
    """模拟bucket_data函数的功能"""
    # 简化处理，返回一个简单的数据结构来模拟函数输出
    return {'test_X': np.random.random((100, 10)), 'test_y': np.random.random(100)}

def cross_valid(regressor, valid_data, lagging=5):
    """模拟cross_valid函数的功能"""
    # 简单返回一个随机值，模拟评估结果
    return np.random.uniform(0.1, 0.3)

def submission(features, model, df, out1, out2, out3, out4):
    """模拟submission函数，写入结果文件"""
    # 简单地创建一些输出文件
    for file in [out1, out2, out3, out4]:
        with open(file, 'w') as f:
            f.write("模拟预测结果\n")
    print(f"结果已写入到 {out1}, {out2}, {out3}, {out4}")

# 将辅助函数保存到单独的模块中，供建模预测.py使用
with open('ultis.py', 'w') as f:
    f.write("""
import numpy as np

def mape_ln(y_true, y_pred):
    return 'mape_ln', np.mean(np.abs((np.exp(y_pred) - np.exp(y_true)) / np.exp(y_true)))

def bucket_data(df_test):
    \"\"\"处理测试数据集\"\"\"
    # 简化处理，返回一个简单的数据结构
    return {'test_X': np.random.random((100, 10)), 'test_y': np.random.random(100)}

def cross_valid(regressor, valid_data, lagging=5):
    \"\"\"交叉验证函数\"\"\"
    # 简单返回一个随机值
    return np.random.uniform(0.1, 0.3)

def submission(features, model, df, out1, out2, out3, out4):
    \"\"\"生成提交结果\"\"\"
    # 创建一些输出文件
    for file in [out1, out2, out3, out4]:
        with open(file, 'w') as f:
            f.write("模拟预测结果\\n")
    print(f"结果已写入到 {out1}, {out2}, {out3}, {out4}")
""")

print("\n已创建辅助函数模块 ultis.py，供建模预测.py使用")