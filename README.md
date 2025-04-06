# UrbanFlowCast

基于XGBoost的智慧交通通行时间预测系统，用于预测城市道路的实时通行情况。

## 项目概述

UrbanFlowCast 是一个利用机器学习技术预测城市道路通行时间的开源项目。通过分析历史交通数据和各种影响因素，该系统能够为出行规划和交通管理提供准确的预测。

### 主要特点

- 基于XGBoost算法的高精度预测模型
- 详细的数据预处理和特征工程
- 时间序列滞后特征分析
- 交叉验证评估模型性能
- 生成多种评估指标的预测结果

## 安装指南

### 环境要求

- Python 3.8+
- 相关Python包（见requirements.txt）

### 安装步骤

1. 克隆仓库到本地
```bash
git clone https://github.com/yourusername/UrbanFlowCast.git
cd UrbanFlowCast
```

2. 创建并激活虚拟环境（可选但推荐）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖包
```bash
pip install -r requirements.txt
```

## 使用方法

### 数据预处理

运行特征提取和数据预处理脚本：

```bash
python 交通数据预处理-特征提取.py
```

这将处理原始交通数据并生成训练所需的特征。如果没有提供原始数据，脚本会自动生成模拟数据用于演示。

### 模型训练与预测

训练模型并生成预测结果：

```bash
python 建模预测.py
```

这将训练XGBoost模型并保存到`model`目录下，同时将预测结果保存到`submission`目录。

## 文件结构

```
UrbanFlowCast/
│
├── 交通数据预处理-特征提取.py  # 数据预处理和特征工程
├── 建模预测.py                # 模型训练和预测生成
├── ultis.py                  # 辅助函数和工具
│
├── data/                     # 数据目录
│   └── training.txt         # 处理后的训练数据
│
├── model/                    # 模型保存目录
│   └── xgbr.model           # 训练好的XGBoost模型
│
├── submission/               # 预测结果目录
│   ├── xgbr1.txt
│   ├── xgbr2.txt
│   ├── xgbr3.txt
│   └── xgbr4.txt
│
└── figures/                  # 可视化结果目录
    ├── feature_importance.png
    ├── lagged_features_correlation.png
    └── travel_time_heatmap.png
```

## 模型说明

本项目使用XGBoost（极致梯度提升）算法构建预测模型，主要特点：

1. **特征工程**：
   - 时间特征（小时、分钟、星期几等）
   - 滞后特征（前几个时间段的通行情况）
   - 统计特征（不同时间段的平均值、标准差等）

2. **模型训练**：
   - 使用5折时间序列交叉验证
   - 针对不同时间段单独评估模型性能
   - 调整超参数以优化预测结果

3. **评估指标**：
   - 使用MAPE-LN（对数平均绝对百分比误差）作为主要评估指标

## 贡献指南

欢迎通过以下方式为项目做出贡献：

1. Fork 仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请查看 [LICENSE](LICENSE) 文件

## 联系方式

项目作者：xdis - 1002860741@qq.com
