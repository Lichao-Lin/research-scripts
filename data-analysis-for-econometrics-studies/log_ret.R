# 加载必要的包
library(tseries)
library(psych)
library(knitr)
library(kableExtra)
library(dplyr)

# 读取CSV文件（请修改为您的文件路径）
data <- read.csv("your_file_path.csv")

# 假设数据有两列：date 和 close
# 将date列转换为日期格式
data$date <- as.Date(data$date)

# 按日期排序（确保时间顺序）
data <- data %>% arrange(date)

# 计算对数收益率（百分比形式）
log_ret <- diff(log(data$close)) * 100

# 基本统计量
mean_ret <- mean(log_ret)
var_ret <- var(log_ret)
skew <- skewness(log_ret)
kurt <- kurtosis(log_ret) - 3  # 超额峰度

# Jarque-Bera检验
jb <- jarque.bera.test(log_ret)

# Ljung-Box检验对平方残差
squared_ret <- log_ret^2
lb5 <- Box.test(squared_ret, lag = 5, type = "Ljung-Box")
lb10 <- Box.test(squared_ret, lag = 10, type = "Ljung-Box")
lb15 <- Box.test(squared_ret, lag = 15, type = "Ljung-Box")

# ADF检验
# 无常数项
adf_no_const <- adf.test(log_ret, k = 0, alternative = "stationary")
# 有漂移项（常数项）
adf_with_drift <- adf.test(log_ret, k = 0, alternative = "stationary")
# 有漂移项和趋势项
adf_with_trend <- adf.test(log_ret, k = 0, alternative = "stationary")

# 获取p值
skew_p <- 2 * (1 - pnorm(abs(skew) / sqrt(6/length(log_ret))))
kurt_p <- 2 * (1 - pnorm(abs(kurt) / sqrt(24/length(log_ret))))
mean_ret_p <- 2 * (1 - pnorm(abs(mean_ret) / sqrt(var_ret/length(log_ret))))

# 构建表格数据
results <- data.frame(
  统计量 = c("Mean", "Variance", "Skewness", "Excess kurtosis", 
            "Jarque-Bera test", 
            "Ljung-Box statistics Q^2(m) to the squared residuals",
            "Q^2(5)", "Q^2(10)", "Q^2(15)",
            "Augmented Dickey-Fuller (ADF) test",
            "ADF- no constant term", "ADF- with drift", "ADF- with drift and trend"),
  估计值 = c(sprintf("%.4f", mean_ret), 
            sprintf("%.4f", var_ret),
            sprintf("%.4f***", skew), 
            sprintf("%.4f***", kurt),
            sprintf("%.4f***", jb$statistic),
            "",  # 占位
            sprintf("%.4f***", lb5$statistic),
            sprintf("%.4f***", lb10$statistic),
            sprintf("%.4f***", lb15$statistic),
            "",
            sprintf("%.4f***", adf_no_const$statistic),
            sprintf("%.4f***", adf_with_drift$statistic),
            sprintf("%.4f***", adf_with_trend$statistic)),
  P值 = c(sprintf("(%.4f)", mean_ret_p), 
          "", 
          sprintf("(%.4f)", skew_p), 
          sprintf("(%.4f)", kurt_p),
          sprintf("(%.4f)", jb$p.value),
          "",
          sprintf("(%.4f)", lb5$p.value),
          sprintf("(%.4f)", lb10$p.value),
          sprintf("(%.4f)", lb15$p.value),
          "",
          sprintf("(%.4f)", adf_no_const$p.value),
          sprintf("(%.4f)", adf_with_drift$p.value),
          sprintf("(%.4f)", adf_with_trend$p.value))
)

# 输出三线表（控制台显示）
print(results)

# 如果需要保存为图片，可以使用以下代码
# 保存为CSV
write.csv(results, "summary_statistics.csv", row.names = FALSE)

# 或者使用kableExtra生成HTML表格并截图
kable(results, format = "html", align = "c", 
      caption = "表1 S&P500指数日度收益率统计摘要") %>%
  kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = FALSE, 
                position = "center") %>%
  row_spec(0, bold = TRUE) %>%
  save_kable("table1.html", self_contained = TRUE)

