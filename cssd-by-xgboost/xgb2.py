


# 基础库
# --------------------------
import os             
import pandas as pd   
import matplotlib.pyplot as plt  
import shap
import numpy as np
# -------------------------
from xgboost import XGBRegressor, plot_importance

# --------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --------------------------
cssd = pd.read_csv('C:\\Users\\megamind\\Desktop\\crash_a.csv', parse_dates=['Trddt'])


# --------------------------
#  排序（保证时间顺序）
# --------------------------
cssd = cssd.sort_values('Trddt').reset_index(drop=True)

# --------------------------
# 4. 构造滞后特征（Lags）
# --------------------------
# 这里构造 CSSD1 的1天、2天、5天滞后
cssd['CSSD1_lag1'] = cssd['CSSD1'].shift(1)
cssd['CSSD1_lag2'] = cssd['CSSD1'].shift(2)
cssd['CSSD1_lag5'] = cssd['CSSD1'].shift(5)

# 构造简单的滚动平均作为特征
cssd['CSSD1_ma5'] = cssd['CSSD1'].rolling(5).mean()
cssd['CSSD1_ma10'] = cssd['CSSD1'].rolling(10).mean()

# --------------------------
# 删除含NaN的行（滞后期开头会产生NaN）
# --------------------------
cssd = cssd.dropna().reset_index(drop=True)

# --------------------------
# 6. 特征和目标
# --------------------------
# 特征选择: 滞后期 + 移动平均
feature_cols = ['CSSD1', 'CSSD1_lag1', 'CSSD1_lag2', 'CSSD1_lag5', 'CSSD1_ma5', 'CSSD1_ma10']
X = cssd[feature_cols]

# 预测目标: 下一个交易日的市场回报
y = cssd['Dretwdeq'].shift(-1)  # 向前滚动一位预测下一天
cssd = cssd.iloc[:-1]  # 删除最后一行的NaN
X = X.iloc[:-1]
y = y.iloc[:-1]

# --------------------------
# 4. 滚动预测设置
# --------------------------
window_size = 240     # 滚动训练窗口长度
preds = []
feat_importances = pd.DataFrame(columns=feature_cols)

for start in range(len(cssd) - window_size):
    end = start + window_size
    X_train = X.iloc[start:end]
    y_train = y.iloc[start:end]
    X_test = X.iloc[end:end+1]
    y_test = y.iloc[end:end+1]

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    preds.append(pred)

    # 记录特征重要性
    imp = model.feature_importances_
    feat_importances.loc[end] = imp

# --------------------------
# 5. 输出预测值
# --------------------------
cssd = cssd.iloc[window_size:].copy()
cssd['Dretwdeq'] = preds

cssd.to_csv("C:\\Users\\megamind\\Desktop\\3cssd_prediction.csv", index=False)
print("滚动预测完成，文件已保存为 cssd_prediction.csv")



# 1. 创建保存文件夹
# --------------------------
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_dir = os.path.join(desktop_path, "xgb_visualizations")
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# 2. 特征重要性可视化
# --------------------------
final_model = model  # 循环里最后训练的模型
features = feature_cols

# weight
plt.figure(figsize=(8,6))
plot_importance(final_model, importance_type='weight', title='Feature Importance (weight)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_weight.png'))
plt.close()

# gain
plt.figure(figsize=(8,6))
plot_importance(final_model, importance_type='gain', title='Feature Importance (gain)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_gain.png'))
plt.close()

# cover
plt.figure(figsize=(8,6))
plot_importance(final_model, importance_type='cover', title='Feature Importance (cover)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_cover.png'))
plt.close()

# --------------------------
# 3. 残差可视化
# --------------------------
y_true = y.iloc[window_size:].values
y_pred = preds
residuals = y_true - y_pred

plt.figure(figsize=(8,4))
plt.plot(residuals, label='Residuals')
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals (y_true - y_pred)')
plt.xlabel('Time Index')
plt.ylabel('Residual')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residual_plot.png'))
plt.close()

# 1. 选择用于解释的样本（滚动预测对应区间）
X_explain = X.iloc[window_size:window_size+500].copy()

# 2. 初始化 SHAP TreeExplainer（XGBoost 原生支持）
explainer = shap.TreeExplainer(final_model)

# --------------------------
# 一、SHAP 主效应分析
# --------------------------

# 1. 计算 SHAP values
shap_values = explainer.shap_values(X_explain)

# 2. SHAP 特征重要性（mean |SHAP|）
shap_importance = (
    pd.DataFrame({
        "feature": X_explain.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    })
    .sort_values("mean_abs_shap", ascending=False)
)

# 3. 保存表格
shap_importance.to_csv(
    os.path.join(output_dir, "shap_feature_importance.csv"),
    index=False
)

# 4. SHAP summary bar plot
plt.figure()
shap.summary_plot(
    shap_values,
    X_explain,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_importance_bar.png"), dpi=300)
plt.close()

# --------------------------
# 二、SHAP 交互效应分析
# --------------------------

# 1. 计算 SHAP 交互值
shap_interaction_values = explainer.shap_interaction_values(X_explain)

# 2. 计算交互强度（平均绝对值）
interaction_strength = np.abs(shap_interaction_values).mean(axis=0)

# 3. 整理成 DataFrame（只取上三角，避免重复）
features = X_explain.columns
records = []

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        records.append({
            "feature_1": features[i],
            "feature_2": features[j],
            "interaction_strength": interaction_strength[i, j]
        })

interaction_df = (
    pd.DataFrame(records)
    .sort_values("interaction_strength", ascending=False)
)

# 4. 保存交互排名表
interaction_df.to_csv(
    os.path.join(output_dir, "shap_interaction_ranking.csv"),
    index=False)
