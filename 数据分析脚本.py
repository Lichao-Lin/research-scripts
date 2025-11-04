import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor, reset_ramsey
from linearmodels.iv import IV2SLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

# ------------------------- 功能函数 -------------------------
def base_regression(X, y, df):
    """ 功能1：基础回归分析 """
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print("\n" + "="*50)
    print(f"基础回归结果（因变量：{df.columns[y.name]})")
    print("="*50)
    print(model.summary())
    return model

def white_test(model):
    """ 功能2：White异方差检验 """
    test_result = het_white(model.resid, model.model.exog)
    print("\nWhite检验结果：")
    print(f"LM统计量 = {test_result[0]:.4f}")
    print(f"p值 = {test_result[1]:.4f}")

def wls_regression(X, y, weights):
    """ 加权最小二乘法 """
    X = sm.add_constant(X)
    model = sm.WLS(y, X, weights=weights).fit()
    print("\nWLS回归结果：")
    print(model.summary())
    return model

def reset_test(model, degree=3):
    """ 功能3：RESET模型设定检验 """
    test_result = reset_ramsey(model, degree=degree)
    print(f"\nRESET检验（{degree}阶）：")
    print(f"F统计量 = {test_result.statistic:.4f}")
    print(f"p值 = {test_result.pvalue:.4f}")

def vif_calculation(X):
    """ 计算方差膨胀因子 """
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\n方差膨胀因子(VIF)：")
    for col, v in zip(X.columns, vif):
        print(f"{col:>15}: {v:.2f}")

def iv_regression(df, endog_var, instruments, exog_vars, y_col):
    """ 功能4：工具变量回归 """
    formula = f"{y_col} ~ 1 + {exog_vars} + [{endog_var} ~ {instruments}]"
    model = IV2SLS.from_formula(formula, data=df).fit()
    print("\n2SLS回归结果：")
    print(model.summary)

def adf_test(series):
    """ 功能5：ADF单位根检验 """
    result = adfuller(series.dropna())
    print("\nADF检验结果：")
    print(f"ADF统计量：{result[0]:.4f}")
    print(f"p值：{result[1]:.4f}")
    print("临界值：")
    for key, value in result[4].items():
        print(f"{key}: {value:.4f}")

# ------------------------- 主程序 -------------------------
def main():
    # 读取数据
    file_path = input("请输入Excel文件路径：").strip()
    try:
        df = pd.read_excel(file_path).apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print("文件读取失败：", str(e))
        return
    
    # 显示数据列
    print("\n[数据列]")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}: {col}")
    
    # 功能菜单
    menu = """
    === 计量经济学分析工具 ===
    1. 基础回归分析
    2. 异方差检验与处理
    3. 模型设定检验(VIF/RESET)
    4. 工具变量回归(2SLS)
    5. 时间序列分析(ADF/VAR)
    6. 退出
    """
    
    current_model = None  # 保存当前模型结果
    
    while True:
        print(menu)
        choice = input("请输入功能编号(1-6)：").strip()
        
        if choice == '6':
            print("程序已退出")
            break
            
        try:
            # 公共数据准备
            if choice in ['1','2','3','4']:
                y_col = int(input("请输入因变量列号：")) - 1
                x_cols = [int(x)-1 for x in input("请输入自变量列号（空格分隔）：").split()]
                X = df.iloc[:, x_cols].dropna(axis=1)
                y = df.iloc[:, y_col]
                data = pd.concat([X, y], axis=1).dropna()
                X_clean = data.iloc[:, :-1]
                y_clean = data.iloc[:, -1]

            # 功能路由
            if choice == '1':
                current_model = base_regression(X_clean, y_clean, df)
                
            elif choice == '2':
                if current_model is None:
                    current_model = base_regression(X_clean, y_clean, df)
                white_test(current_model)
                
                # WLS处理
                if input("是否进行WLS修正？(y/n)：").strip().lower() == 'y':
                    weights = 1 / np.abs(current_model.resid)
                    current_model = wls_regression(X_clean, y_clean, weights)
                    
            elif choice == '3':
                if current_model is None:
                    current_model = base_regression(X_clean, y_clean, df)
                reset_test(current_model)
                vif_calculation(X_clean)
                
            elif choice == '4':
                endog_var = df.columns[int(input("内生变量列号："))-1]
                instruments = df.columns[int(input("工具变量列号："))-1]
                exog_vars_list = [df.columns[x] for x in x_cols if df.columns[x] != endog_var]
                exog_vars = "+".join(exog_vars_list) if exog_vars_list else "1"
                iv_regression(df, endog_var, instruments, exog_vars, df.columns[y_col])
                
            elif choice == '5':
                ts_col = int(input("时间序列列号：")) - 1
                series = df.iloc[:, ts_col]
                adf_test(series)
                
                if input("是否建立VAR模型？(y/n)：").strip().lower() == 'y':
                    var_cols = [int(x)-1 for x in input("请输入VAR变量列号（至少2个，空格分隔）：").split()]
                    var_data = df.iloc[:, var_cols].dropna()
                    var_model = VAR(var_data).fit(maxlags=2)
                    print(var_model.summary())
                    
            else:
                print("无效的编号，请重新输入")
                
        except Exception as e:
            print(f"发生错误：{str(e)}")
            continue

if __name__ == "__main__":
    main()
