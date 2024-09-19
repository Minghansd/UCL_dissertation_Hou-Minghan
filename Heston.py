import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Heston 模型特征函数
def heston_char_func(phi, S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
    x = np.log(S0 / K)
    a = kappa * theta
    u = 0.5
    b = kappa + rho * sigma * phi * 1j if option_type == 'call' else kappa - rho * sigma * phi * 1j
    d = np.sqrt(b * b - sigma ** 2 * (2 * u * phi * 1j - phi ** 2))
    g = (b - d) / (b + d)
    C = r * phi * 1j * T + a / (sigma ** 2) * (
        (b - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = (b - d) / (sigma ** 2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    return np.exp(C + D * v0 + 1j * phi * x)

# Heston 模型期权定价公式
def heston_price(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
    P1 = lambda phi: (np.exp(-1j * phi * np.log(K)) * heston_char_func(phi - 1j, S0, K, T, r, kappa, theta, sigma, rho, v0, option_type) /
                      (1j * phi)).real
    P2 = lambda phi: (np.exp(-1j * phi * np.log(K)) * heston_char_func(phi, S0, K, T, r, kappa, theta, sigma, rho, v0, option_type) /
                      (1j * phi)).real
    int1 = 0.5 + (1 / np.pi) * quad(P1, 0, 100)[0]
    int2 = 0.5 + (1 / np.pi) * quad(P2, 0, 100)[0]
    if option_type == 'call':
        return S0 * int1 - np.exp(-r * T) * K * int2
    elif option_type == 'put':
        return np.exp(-r * T) * K * (1 - int2) - S0 * (1 - int1)

# 读取数据
file_path = 'data/BS.xlsx'
data = pd.read_excel(file_path)
data.fillna(method='ffill', inplace=True)

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

# Heston 模型参数 (这些参数可以根据需要调整)
kappa = 2.0     # 速度参数
theta = 0.02    # 长期方差
rho = -0.5      # 波动率和资产价格的相关系数
v0 = 0.01       # 初始方差

# 使用Heston模型计算期权的理论价格（假设是看涨期权）
predicted_theoretical_prices = np.array([
    heston_price(S[i], K[i], T[i], r[i], kappa, theta, sigma[i], rho, v0, option_type='call')
    for i in range(len(S))
])

# 将计算得到的理论价格与实际理论价格对比
result_df = pd.DataFrame({
    'Real Closing Price': real_theoretical_price.flatten(),
    'Predicted Closing Price': predicted_theoretical_prices.flatten()
})

# 计算评价指标
rmse = np.sqrt(mean_squared_error(real_theoretical_price, predicted_theoretical_prices))
mae = mean_absolute_error(real_theoretical_price, predicted_theoretical_prices)
r2 = r2_score(real_theoretical_price, predicted_theoretical_prices)

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
result_file_path = f'{result_save_dir}/Heston_model_theoretical_price_predictions.xlsx'
result_df.to_excel(result_file_path, index=False)
print(f'Results saved to {result_file_path}')

# 绘制真实值与预测值对比图
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(real_theoretical_price, color='blue', label='Real Theoretical Price')
plt.plot(predicted_theoretical_prices, color='red', label='Predicted Theoretical Price (Heston)')
plt.title('Heston Model Theoretical Price Prediction vs Real Theoretical Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig(f'{figure_save_dir}/Heston_Model_Theoretical_Price_Prediction.png', dpi=300)
plt.show()
