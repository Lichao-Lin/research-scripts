# 科研脚本

个人科研与学习脚本，包含计量分析、机器学习、文本分析和实验代码。原四个仓库按目录合并，保留文件与提交历史。

## 目录与文件关系

| 目录 | 内容 |
| --- | --- |
| [cssd-by-xgboost](cssd-by-xgboost/) | CSSD 与市场收益研究。`ols.py` 做线性回归；`xgb2.py` 做 XGBoost 滚动预测和 SHAP 解释。PNG 和 CSV 是已有的特征重要性、残差和交互效应结果。两个脚本可分别运行。 |
| [contribution_matrices](contribution_matrices/) | 文献摘要的关键词与共现网络。`sheet1_col3_top_keywords.py` 辅助选词 → `compute_term_tfidf.py` 计算词权重 → `build_contribution_matrices.py` 生成贡献矩阵 → `visualize_contribution_matrix.py` 绘制网络图。 |
| [data-analysis-for-econometrics-studies](data-analysis-for-econometrics-studies/) | 计量经济学练习。`OLS regression.py` 做回归，`interactive data analysis  script.py` 提供交互式分析，`Stock data visualization.py` 获取并导出股票数据，`multi-method stock data analysis.py` 做时间序列与波动率分析；`log_ret.R`、`r1.R` 是 R 分析脚本。 |
| [Ars-Informatica](Ars-Informatica/) | 早期实验代码。中文命名的 OLS、数据分析和股票分析脚本与计量目录中的同类脚本相对应；`lda.ipynb` 是 LDA 实验，`capm time simulation.m` 是 CAPM 模拟，`MOVING HREAT.cpp` 和同名 EXE 是独立 C++ 小程序。 |

四个目录彼此独立，没有统一运行顺序。`Ars-Informatica` 与计量目录中功能相近的版本均保留，方便对照；其 `.github/` 是原仓库配置的留档。

## 使用

进入所需子目录，选择 Python、R、MATLAB 或 C++ 环境。例如：

```bash
cd cssd-by-xgboost
python ols.py
```

运行前按需安装脚本导入的依赖，并修改数据路径、输出路径和接口配置。原始研究数据需自行准备，现有结果文件供参考。文本分析参数见 [contribution_matrices/README.md](contribution_matrices/README.md)。
