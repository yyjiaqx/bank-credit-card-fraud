# 💳 信用卡欺诈交易预测

**Credit Card Fraud Detection** — 基于可解释特征的银行信用卡欺诈检测。

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44+-red.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

对比 **Logistic Regression / Random Forest / XGBoost** 三种模型识别信用卡欺诈交易。使用 SMOTE 处理极端不平衡（欺诈仅 1.51%），GridSearchCV 调优，SHAP 可解释性分析。数据集包含**真实业务特征**（交易金额、商户类别、交易时段、设备信任分等），模型结果可直接转化为风控规则。

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
| `transaction_id` | 交易ID | 删除 |
| `amount` | 金额 | StandardScaler |
| `transaction_hour` | 交易时段(0-23) | sin/cos 循环编码 |
| `merchant_category` | 商户类别(5类) | OneHotEncoder |
| `foreign_transaction` | 是否境外 | 保持 |
| `location_mismatch` | 位置是否不匹配 | 保持 |
| `device_trust_score` | 设备信任分(25-99) | StandardScaler |
| `velocity_last_24h` | 24h交易次数 | StandardScaler |
| `cardholder_age` | 持卡人年龄 | StandardScaler |
| **`is_fraud`** | **目标标签** | — |

## 🏗️ 项目结构

```
credit-card-fraud-detection/
├── data/
│   └── credit_card_fraud_10k.csv
├── notebooks/
│   └── fraud_detection.ipynb
├── src/
│   ├── data_loader.py           ← 数据加载 + EDA
│   ├── preprocessing.py         ← 特征工程 + SMOTE
│   ├── models.py                ← LR/RF/XGBoost + SHAP
│   └── evaluation.py            ← 多模型评估对比
├── outputs/                     ← 12+ 张分析图表
├── main.py                      ← 一键运行全管线
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 快速开始

```bash
cd credit-card-fraud-detection
pip install -r requirements.txt
python main.py
```

## 🧠 方法

### 特征工程
- `merchant_category` → OneHot 编码（5类 → 5个二值列）
- `transaction_hour` → sin/cos 循环编码（保留周期性）
- 数值特征 → StandardScaler 标准化

### 类别平衡
- **SMOTE** 过采样，将欺诈:正常从 ~1:65 平衡至 ~1:1

### 模型
- **Logistic Regression**（基线）— GridSearchCV 调 C
- **Random Forest** — 调 n_estimators/max_depth
- **XGBoost** — 调 n_estimators/max_depth/learning_rate

### 可解释性
- **特征重要性** — RF/XGBoost 的 feature_importances_
- **SHAP** — Summary/Bar/Waterfall/Dependence 四图，单笔交易欺诈概率拆解

### 评估
- ROC-AUC / PR-AUC / Recall / Precision / F1-Score
- 混淆矩阵、ROC/PR 曲线叠加对比
- 模型指标柱状图对比

## 📈 结果解读

### 模型性能对比

| 模型 | Recall | Precision | F1-Score | ROC-AUC | PR-AUC |
|------|--------|-----------|----------|---------|--------|
| Logistic Regression | 99.24% | 96.45% | 97.82% | 99.37% | 98.63% |
| Random Forest | **100%** | 98.85% | 99.42% | 100% | 99.96% |
| XGBoost 🏆 | **100%** | **99.85%** | **99.92%** | 100% | 99.98% |

> **XGBoost 最佳：测试集 1,970 笔欺诈零漏判，仅 3 笔误报。**

### 混淆矩阵（XGBoost）

```
              预测正常  预测欺诈
实际正常        1967       3
实际欺诈           0    1970
```

- 真阳性（TP）= 1,970，假阴性（FN）= 0 → **零漏判**
- 假阳性（FP）= 3，真阴性（TN）= 1,967 → 仅 3 笔误报

### 特征重要性 Top 5

| 排名 | 特征 | 说明 |
|------|------|------|
| 1 | device_trust_score | **最重要**，设备信任分低 = 高风险 |
| 2 | foreign_transaction | 境外交易风险更高 |
| 3 | amount | 大额交易需关注 |
| 4 | location_mismatch | 位置不匹配是危险信号 |
| 5 | cardholder_age | 年龄有一定参考价值 |

### 业务建议

1. **设备信任分低 + 境外交易 + 位置不匹配** → 可直接作为风控规则触发人工审核
2. **大额境外交易 + 24h 内高频** → 自动拦截 + 短信验证
3. **SHAP Waterfall 图** → 对每笔可疑交易解释"为什么判定为欺诈"，满足监管可解释性要求

## 🔧 依赖

```
numpy>=1.14.0
pandas>=0.23.0
scikit-learn>=0.24.0
xgboost>=1.5.0
matplotlib>=2.2.0
seaborn>=0.8.0
imbalanced-learn>=0.9.0
shap>=0.44.0
jupyter>=1.0.0
```

## 📝 简历描述参考

> 独立完成信用卡欺诈检测项目，对 10,000 笔交易数据进行特征工程（OneHot编码、循环时间编码）和不平衡处理（SMOTE），对比 Logistic Regression、Random Forest、XGBoost 三种模型并 GridSearchCV 调优，XGBoost 召回率 100%、精准率 99.85%，仅误判 3 笔，结合特征重要性与 SHAP 进行可解释性分析，输出可直接落地的风控规则建议。

## 📄 License

MIT
