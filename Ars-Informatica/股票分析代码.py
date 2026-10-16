import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import sys

# 跨平台设置终端颜色（绿色字体，黑色背景）
if sys.platform.startswith('win'):
    import os
    os.system('color 0A')  # Windows 生效
else:
    print("\033[92m\033[40m")  # macOS/Linux ANSI转义码

# 1. 设置 Tushare Token
ts.set_token('9ee351c4ee776159e070439ba90695695928b260cec1a55f25ebf917')
pro = ts.pro_api()

# 2. 获取一年日线数据
df = pro.daily(ts_code='002920.SZ', start_date='20230101', end_date='20231231')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df.set_index('trade_date', inplace=True)
df.sort_index(inplace=True)
returns = df['close'].pct_change().dropna() * 100  # 日收益率（百分比）

# 3. ADF平稳性检验（完整输出）
def adf_analysis(series):
    adf_result = adfuller(series)
    print("\033[92m" + "="*50 + "\033[0m")
    print("\033[92mADF 平稳性检验结果:\033[0m")
    print(f"\033[92mADF Statistic: {adf_result[0]:.4f}\033[0m")
    print(f"\033[92mp-value: {adf_result[1]:.4f}\033[0m")
    print("\033[92m临界值 (Critical Values):\033[0m")
    for key, value in adf_result[4].items():
        print(f"\033[92m   {key}: {value:.4f}\033[0m")
    print("\033[92m结论：p-value < 0.05 时序列平稳\033[0m\n")
    
    # 绘制ADF检验可视化
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='日收益率', color='dodgerblue')
    plt.axhline(series.mean(), color='red', linestyle='--', label='均值')
    plt.axhline(series.mean() + series.std(), color='grey', linestyle=':', label='±1标准差')
    plt.axhline(series.mean() - series.std(), color='grey', linestyle=':')
    plt.title('ADF检验 - 收益率序列平稳性分析', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('收益率 (%)', fontsize=12)
    plt.legend()
    plt.show()

adf_analysis(returns)

# 4. 自相关检验（ACF/PACF数值+图表）
print("\033[92m" + "="*50 + "\033[0m")
print("\033[92m自相关检验 (ACF/PACF):\033[0m")

# 计算并输出ACF/PACF数值
acf_values = acf(returns, nlags=20)
pacf_values = pacf(returns, nlags=20, method='ywm')
print("\033[92mACF 值（滞后1~5期）:\033[0m")
print(np.round(acf_values[1:6], 4))
print("\033[92mPACF 值（滞后1~5期）:\033[0m")
print(np.round(pacf_values[1:6], 4))

# 绘制ACF/PACF图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(returns, lags=20, ax=ax1, title='自相关函数 (ACF)', color='green')
plot_pacf(returns, lags=20, ax=ax2, title='偏自相关函数 (PACF)', method='ywm', color='green')
plt.show()

# 5. GARCH(1,1)建模（完整参数+可视化）
print("\033[92m" + "="*50 + "\033[0m")
print("\033[92mGARCH(1,1) 模型结果:\033[0m")

# 模型拟合
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(update_freq=5, disp='off')

# 输出完整模型摘要
print("\033[92m模型参数:\033[0m")
print(garch_fit.summary())

# 绘制波动率拟合效果
plt.figure(figsize=(12, 6))
plt.plot(returns, color='blue', alpha=0.4, label='日收益率')
plt.plot(garch_fit.conditional_volatility, color='darkorange', label='GARCH波动率')
plt.title('GARCH(1,1) 模型 - 条件波动率拟合', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('波动率 (%)', fontsize=12)
plt.legend()
plt.show()

# 输出关键参数（绿色字体）
print("\033[92m关键参数:\033[0m")
print(f"\033[92mAlpha (ARCH项): {garch_fit.params['alpha[1]']:.4f}\033[0m")
print(f"\033[92mBeta (GARCH项): {garch_fit.params['beta[1]']:.4f}\033[0m")
print(f"\033[92mOmega (常数项): {garch_fit.params['omega']:.4f}\033[0m")
print(f"\033[92m对数似然值: {garch_fit.loglikelihood:.2f}\033[0m")
print(f"\033[92mAIC准则: {garch_fit.aic:.2f}\033[0m")
