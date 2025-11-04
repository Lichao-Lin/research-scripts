# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:36:56 2025

@author: asus
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

def main():
    # 读取Excel文件
    file_path = input("请输入Excel文件路径：")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print("文件读取失败：{}".format(str(e)))
        return

    # 显示数据预览（带Excel列号）
    print("\n[数据预览] 列号对应关系：")
    print("Excel列号 | 列名")
    print("-------------------")
    for i, col in enumerate(df.columns, 1):
        print("   {:2d}    | {}".format(i, col))  # 修改此处
    print("\n前5行数据：")
    print(df.head())

    # 获取用户输入
    try:
        y_col = int(input("\n请输入因变量的Excel列号（单个数字）：")) - 1
        x_cols = [int(x)-1 for x in input("请输入自变量的Excel列号（多个用空格分隔）：").split()]
    except ValueError:
        print("错误：请输入有效的数字列号")
        return

    # 验证列号有效性
    max_col = len(df.columns) - 1
    if y_col < 0 or y_col > max_col or any(x < 0 or x > max_col for x in x_cols):
        print("错误：列号应在1-{}之间".format(len(df.columns)))  # 修改此处
        return

    # 准备数据
    X = df.iloc[:, x_cols]
    y = df.iloc[:, y_col]
    
    # 处理缺失值
    data = pd.concat([X, y], axis=1).dropna()
    if data.empty:
        print("错误：数据中存在缺失值，删除后无剩余数据")
        return
        
    X_clean = data.iloc[:, :-1]
    y_clean = data.iloc[:, -1]
    
    # 添加常数项
    X_clean = sm.add_constant(X_clean)

    # 执行回归
    try:
        model = sm.OLS(y_clean, X_clean)
        results = model.fit()
    except Exception as e:
        print("回归分析失败：{}".format(str(e)))  # 修改此处
        return

    # 打印结果
    print("\n" + "="*50)
    print("回归分析结果（因变量：{}）".format(df.columns[y_col]))  # 修改此处
    print("="*50)
    print(results.summary())
    
    # 输出关键参数
    print("\n关键参数：")
    print("R-squared: {:.4f}".format(results.rsquared))
    print("Adj R-squared: {:.4f}".format(results.rsquared_adj))
    print("\n系数：")
    for name, coef in zip(X_clean.columns, results.params):
        print("{:>15}: {:.6f}".format(name, coef))  # 修改此处

if __name__ == "__main__":
    main()
