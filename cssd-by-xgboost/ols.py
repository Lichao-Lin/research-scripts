import pandas as pd
import statsmodels.api as sm

# --------------------------
# 1. 读取数据
# --------------------------
cssd = pd.read_csv("C:\\Users\\megamind\\Desktop\\crash_a.csv")


# --------------------------
# 3. 按日期排序
# --------------------------
cssd['Trddt'] = pd.to_datetime(cssd['Trddt'])
cssd = cssd.sort_values('Trddt')

# --------------------------
# 4. 构造下一期回报作为因变量
# --------------------------
cssd['Return_next'] = cssd['Dretwdeq'].shift(-1)

# 删除最后一行 NaN
cssd = cssd.dropna(subset=['Return_next'])

# --------------------------
# 5. 定义自变量 X 和因变量 Y
# --------------------------
X = cssd[['CSSD1']]
Y = cssd['Return_next']

# 添加常数项（截距）
X = sm.add_constant(X)

# --------------------------
# 6. 拟合 OLS 回归
# --------------------------
model = sm.OLS(Y, X)
results = model.fit()

# --------------------------
# 7. 输出回归结果
# --------------------------
print(results.summary())

# --------------------------
# 8. 保存预测值到 CSV
# --------------------------
cssd['Return_pred'] = results.predict(X)
cssd.to_csv("C:\\Users\\megamind\\Desktop\\ols_cssd_predict.csv", index=False)
print("预测值已保存到 ols_cssd_predict.csv")

