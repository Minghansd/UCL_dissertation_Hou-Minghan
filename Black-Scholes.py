import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# B-S模型的实现
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# 读取数据
file_path = 'data/BS.xlsx'
data = pd.read_excel(file_path)
data.fillna(method='ffill', inplace=True)  # 填充数据

# 查看列名，确保选择的列正确
print(data.columns)

# 特征列名（中文到英文映射）
feature_names = {
    "标的收盘价": "Underlying Closing Price",
    "执行价": "Strike Price",
    "距离到期日时间（折算成年）": "Time to Maturity (years)",
    "无风险利率": "Risk-free Rate",
    "经插值的隐含波动率": "Implied Volatility",
    "理论价格": "Theoretical Price"
}

# 提取特征数据（映射到英文）
S = data['标的收盘价'].values  # 标的资产价格
K = data['执行价'].values  # 执行价格
T = data['距离到期日时间（折算成年）'].values  # 距离到期时间（以年为单位）
r = data['无风险利率'].values  # 无风险利率
sigma = data['经插值的隐含波动率'].values  # 波动率
real_theoretical_price = data['理论价格'].values  # 实际理论价格

# 使用B-S模型计算期权的理论价格
predicted_theoretical_prices = black_scholes(S, K, T, r, sigma, option_type='call')

# 将计算得到的理论价格与实际理论价格对比
result_df = pd.DataFrame({
    'Real Closing Price': real_theoretical_price.flatten(),
    'Predicted Closing Price': predicted_theoretical_prices.flatten()
})

# 计算评价指标
rmse = np.sqrt(mean_squared_error(real_theoretical_price, predicted_theoretical_prices))
mae = mean_absolute_error(real_theoretical_price, predicted_theoretical_prices)
r2 = r2_score(real_theoretical_price, predicted_theoretical_prices)

# 打印指标
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R²: {r2}')

# 根据文件名选择保存路径
if 'O510050ivf_20180101至20181231.xlsm' in file_path:
    result_save_dir = 'result1'
    figure_save_dir = 'figure1'
else:
    result_save_dir = 'result2'
    figure_save_dir = 'figure2'

# 创建文件夹（如果不存在）
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)
if not os.path.exists(figure_save_dir):
    os.makedirs(figure_save_dir)

# 保存结果到Excel文件
result_file_path = f'{result_save_dir}/BS_model_theoretical_price_predictions.xlsx'
result_df.to_excel(result_file_path, index=False)
print(f'Results saved to {result_file_path}')

# 绘制真实值与预测值对比图
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(real_theoretical_price, color='blue', label='Real Theoretical Price')
plt.plot(predicted_theoretical_prices, color='red', label='Predicted Theoretical Price (B-S)')
plt.title('B-S Model Theoretical Price Prediction vs Real Theoretical Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig(f'{figure_save_dir}/BS_Model_Theoretical_Price_Prediction.png', dpi=300)
plt.show()
