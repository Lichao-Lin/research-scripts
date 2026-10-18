From a methodological perspective, current research in this field has developed along two complementary paths:

Prediction Path: Using ML models to improve the predictive accuracy of herding behavior and identify important nonlinear factors.

Interpretation Path: Using explainable ML (e.g., SHAP values) to reveal the micro-level determinants of herding behavior and conduct causal inference.

The following code framework primarily follows the second path while incorporating key techniques from the first.

## 预测结果与日期

`xgb2.py` 在当日收盘特征可用后，预测输入中下一交易日的收益。输入需每天一行，包含 `Trddt`、`CSSD1` 和 `Dretwdeq`；脚本里的本地数据、输出路径需按运行环境修改。

| 输出列 | 含义 |
| --- | --- |
| `Trddt` | 特征日期 |
| `Dretwdeq` | 特征日期的原始真实收益，保留不覆盖 |
| `Target_date` | 输入原始时间序列中下一条记录的日期 |
| `Return_next` | `Target_date` 对应的真实收益 |
| `Return_pred` | 对 `Return_next` 的滚动样本外预测 |

残差为 `Return_next - Return_pred`。先构建下一期标签，再删除无效特征行，防止缺失值把目标错误移到更晚日期。窗口默认使用前 240 条有效观测，保留原来的特征和 XGBoost 参数；有效数据必须超过窗口长度。

`forecasting.py` 封装数据准备与滚动预测，`tests/test_forecasting.py` 用合成数据验证目标日期、原值保留、缺失值与训练窗口边界：

```bash
python -m pip install numpy pandas xgboost scikit-learn
python -m unittest discover -s cssd-by-xgboost/tests -v
```

注意：OLS 脚本为全样本拟合；现有 SHAP 图解释的是最后一个滚动模型，可能包含该模型的训练观测，不代表每个预测时点的样本外解释，也不能单独据此认定因果关系。仓库已有 PNG/CSV 是历史结果，未随本次修复重新生成；没有原始研究数据时，测试不验证预测精度或论文结论。
