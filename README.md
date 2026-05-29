# 💳 信用卡欺诈交易预测

**Credit Card Fraud Detection** — 基于可解释特征的银行信用卡欺诈检测。

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

使用机器学习识别信用卡欺诈交易。与常见的 PCA 黑盒数据不同，本数据集包含**可解释的真实业务特征**：交易金额、商户类别、交易时段、是否跨境、设备信任分等，这意味着模型结果可直接转化为风控规则。

**核心目标：** 最大化召回率（Recall），尽可能减少欺诈漏判。

## 📊 数据集

| 指标 | 数值 |
|------|------|
| 总交易数 | 10,000 |
| 欺诈交易 | 151 (1.51%) |
| 正常交易 | 9,849 |
| 原始特征 | 10（全部可解释） |

| 特征 | 说明 | 处理 |
|------|------|------|
| 	ransaction_id | 交易ID | 删除 |
| mount | 金额 | StandardScaler |
| 	ransaction_hour | 交易时段(0-23) | sin/cos 循环编码 |
| merchant_category | 商户类别(5类) | OneHotEncoder |
| oreign_transaction | 是否境外 | 保持 |
| location_mismatch | 位置是否不匹配 | 保持 |
| device_trust_score | 设备信任分(25-99) | StandardScaler |
| elocity_last_24h | 24h交易次数 | StandardScaler |
| cardholder_age | 持卡人年龄 | StandardScaler |
| **is_fraud** | **目标标签** | — |

## 🏗️ 项目结构

`
credit-card-fraud-detection/
├── data/
│   └── credit_card_fraud_10k.csv
├── notebooks/
│   └── fraud_detection.ipynb    ← 核心分析 Notebook
├── src/
│   ├── data_loader.py           ← 数据加载、EDA、相关性分析
│   ├── preprocessing.py         ← 特征工程（OneHot/循环编码/标准化/SMOTE）
│   ├── models.py                ← LR / RF / XGBoost + GridSearchCV
│   └── evaluation.py            ← 评估 + 可视化 + 特征重要性
├── outputs/
├── main.py                      ← 一键运行
├── README.md
├── requirements.txt
└── .gitignore
`

## 🚀 快速开始

`ash
cd credit-card-fraud-detection

# 安装依赖
pip install -r requirements.txt

# 一键运行
python main.py

# 或打开 Notebook
jupyter notebook notebooks/fraud_detection.ipynb
`

## 🧠 方法

### 特征工程
- merchant_category → OneHot 编码（5类 → 5个二值列）
- 	ransaction_hour → sin/cos 循环编码（保留"23点与0点相邻"的周期性）
- 数值特征 → StandardScaler 标准化
- SMOTE 过采样处理不平衡

### 模型
- **Logistic Regression**（基线）
- **Random Forest**（集成方法）
- **XGBoost**（梯度提升）

### 评估
- ROC-AUC / PR-AUC
- Recall、Precision、F1-Score
- 混淆矩阵
- 特征重要性排序

## 📝 简历描述参考

> 独立完成信用卡欺诈检测项目，对 10,000 笔交易数据进行特征工程（OneHot编码、循环时间编码）和不平衡处理（SMOTE），对比逻辑回归、随机森林、XGBoost 三种模型，通过 GridSearchCV 调优，最终模型召回率达 XX%，精准识别欺诈交易并给出可解释的业务建议。

## 📄 License

MIT
