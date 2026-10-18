


# 基础库
# --------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np
# -------------------------
from xgboost import plot_importance

# --------------------------

from forecasting import FEATURE_COLUMNS, prepare_forecast_data, rolling_forecast

# --------------------------
cssd = pd.read_csv('C:\\Users\\megamind\\Desktop\\crash_a.csv', parse_dates=['Trddt'])


# --------------------------
# 按原始交易日期对齐目标，再构造滞后和滚动平均特征。
# --------------------------
cssd = prepare_forecast_data(cssd)

# --------------------------
# 6. 特征和目标
# --------------------------
# 特征选择: 滞后期 + 移动平均
feature_cols = FEATURE_COLUMNS
X = cssd[feature_cols]

# --------------------------
# 4. 滚动预测设置
# --------------------------
window_size = 240     # 滚动训练窗口长度
forecast_output, model, feat_importances = rolling_forecast(cssd, window_size)
preds = forecast_output['Return_pred'].to_numpy()

# --------------------------
# 5. 输出预测值
# --------------------------
cssd = forecast_output  # 保留 Dretwdeq；另存目标日期、真实目标值和预测值

cssd.to_csv("C:\\Users\\megamind\\Desktop\\3cssd_prediction.csv", index=False)
print("滚动预测完成，文件已保存为 3cssd_prediction.csv")



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
y_true = cssd['Return_next'].to_numpy()
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
