
data <- read.csv("C:/Users/asus/Desktop/3-SP500.csv")

colnames(data)
head(data)

date_col <- "observation_date"     
price_col <- "SP500"   
data <- read.csv("C:/Users/asus/Desktop/3-SP500.csv")

# 查看数据结构
cat("=== 数据列名 ===\n")
print(colnames(data))

cat("\n=== 数据前6行 ===\n")
print(head(data))

dates <- as.Date(data[[date_col]])
prices <- data[[price_col]]

valid <- !is.na(prices) & !is.na(dates)
prices <- prices[valid]
dates <- dates[valid]

order_idx <- order(dates)
prices <- prices[order_idx]
dates <- dates[order_idx]

# 对数收益率
log_ret <- diff(log(prices)) * 100

# 基本统计量
n <- length(log_ret)
mean_ret <- mean(log_ret)
var_ret <- var(log_ret)
sd_ret <- sd(log_ret)

# 偏度
skew <- sum((log_ret - mean_ret)^3) / n / (sd_ret^3)

# 超额峰度
kurt <- sum((log_ret - mean_ret)^4) / n / (sd_ret^4) - 3

# p值
skew_se <- sqrt(6/n)
kurt_se <- sqrt(24/n)
skew_p <- 2 * (1 - pnorm(abs(skew) / skew_se))
kurt_p <- 2 * (1 - pnorm(abs(kurt) / kurt_se))

# Jarque-Bera检验
jb_stat <- n * (skew^2/6 + kurt^2/24)
jb_p <- 1 - pchisq(jb_stat, 2)

# Ljung-Box检验函数
lb_test <- function(x, lag) {
  n <- length(x)
  r <- acf(x, lag.max = lag, plot = FALSE)$acf[-1]
  stat <- n * (n + 2) * sum(r^2 / (n - 1:lag))
  p_value <- 1 - pchisq(stat, lag)
  return(c(stat, p_value))
}

squared_ret <- log_ret^2
lb5 <- lb_test(squared_ret, 5)
lb10 <- lb_test(squared_ret, 10)
lb15 <- lb_test(squared_ret, 15)

# 均值p值
mean_p <- 2 * (1 - pnorm(abs(mean_ret) / sqrt(var_ret/n)))

# ADF检验
if (exists("adf.test", mode = "function")) {
  adf_no <- adf.test(log_ret, k = 0)
  adf_no_stat <- adf_no$statistic
  adf_no_p <- adf_no$p.value
} else {
  adf_no_stat <- NA
  adf_no_p <- NA
}

# 结果表格
results <- data.frame(
  统计量 = c(
    "Mean",
    "Variance", 
    "Skewness",
    "Excess kurtosis",
    "Jarque-Bera test",
    "Ljung-Box statistics Q^2(m) to the squared residuals",
    "Q^2(5)",
    "Q^2(10)", 
    "Q^2(15)",
    "Augmented Dickey-Fuller (ADF) test",
    "ADF- no constant term"
  ),
  估计值 = c(
    sprintf("%.4f", mean_ret),
    sprintf("%.4f", var_ret),
    sprintf("%.4f***", skew),
    sprintf("%.4f***", kurt),
    sprintf("%.4f***", jb_stat),
    "",
    sprintf("%.4f***", lb5[1]),
    sprintf("%.4f***", lb10[1]),
    sprintf("%.4f***", lb15[1]),
    "",
    sprintf("%.4f", adf_no_stat)
  ),
  P值 = c(
    sprintf("(%.4f)", mean_p),
    "",
    sprintf("(%.4f)", skew_p),
    sprintf("(%.4f)", kurt_p),
    sprintf("(%.4f)", jb_p),
    "",
    sprintf("(%.4f)", lb5[2]),
    sprintf("(%.4f)", lb10[2]),
    sprintf("(%.4f)", lb15[2]),
    "",
    sprintf("(%.4f)", adf_no_p)
  )
)

# 打印结果
print(results)

# 保存结果
# 保存结果到桌面
desktop_path <- "C:/Users/asus/Desktop/"
write.csv(results, paste0(desktop_path, "summary_statistics.csv"), row.names = FALSE)
cat("\n结果已保存到：", paste0(desktop_path, "summary_statistics.csv"), "\n")

# 保存为文本文件
sink(paste0(desktop_path, "summary_statistics.txt"))
cat("表1 S&P500指数日度收益率统计摘要\n")
cat("========================================\n")
cat(paste0("样本期间：", min(dates), " 至 ", max(dates), "\n"))
cat(paste0("观测值数量：", n, "\n\n"))
print(results)
cat("\n注：***表示在1%水平上显著，**表示在5%水平上显著，*表示在10%水平上显著\n")
sink()

cat("文本结果已保存到：", paste0(desktop_path, "summary_statistics.txt"), "\n")
