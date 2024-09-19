import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取原始数据
file_path = 'data/BS.xlsx'  # 将其替换为你的实际文件路径
original_data = pd.read_excel(file_path)
print(original_data)

# 根据文件名确定保存目录
if 'O510050ivf_20180101至20181231.xlsm' in file_path:
    figure_save_dir = 'figure1'
    result_save_dir = 'result1'
else:
    figure_save_dir = 'figure2'
    result_save_dir = 'result2'

# 创建figure文件夹（如果不存在）
if not os.path.exists(figure_save_dir):
    os.makedirs(figure_save_dir)
# 创建result文件夹（如果不存在）
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)

# 读取每个模型的预测结果
files = {
    'ANN': f'{result_save_dir}/ANN_closing_price_predictions.xlsx',
    'BS': f'{result_save_dir}/BS_model_theoretical_price_predictions.xlsx',
    'CNN': f'{result_save_dir}/CNN_closing_price_predictions.xlsx',
    'Heston': f'{result_save_dir}/Heston_model_theoretical_price_predictions.xlsx',
    'LSTM': f'{result_save_dir}/LSTM_closing_price_predictions.xlsx'
}

# 加载数据
predictions = {}
for model_name, file_path in files.items():
    predictions[model_name] = pd.read_excel(file_path)

# 合并`距离到期日天数`与`报价时间`到预测结果
for model_name, df in predictions.items():
    if model_name in ['BS', 'Heston']:
        df['Maturity'] = original_data['距离到期日天数']
        df['Date'] = original_data['报价时间']
    else:
        df['Maturity'] = original_data['距离到期日天数'][-len(df):].values
        df['Date'] = original_data['报价时间'][-len(df):].values

# 计算每个模型的RMSE和MAE
metrics = {}
for model_name, df in predictions.items():
    print(model_name)
    df['Error'] = df['Real Closing Price'] - df['Predicted Closing Price']
    df['Squared Error'] = df['Error'] ** 2
    df['Absolute Error'] = df['Error'].abs()

    metrics[model_name] = df.groupby('Maturity').agg({
        'Squared Error': 'mean',
        'Absolute Error': 'mean'
    }).reset_index()

# 生成按Maturity分组的RMSE和MAE对比图
plt.figure(figsize=(10, 6))

# Plot RMSE
for model_name, metric_df in metrics.items():
    plt.plot(metric_df['Maturity'], np.sqrt(metric_df['Squared Error']), label=f'{model_name}_RMSE')

plt.title('Comparison of RMSE (sorted by Maturity)')
plt.xlabel('Maturity')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig(f'{figure_save_dir}/comparison_rmse_maturity.png', dpi=300)
plt.show()

# Plot MAE
plt.figure(figsize=(10, 6))

for model_name, metric_df in metrics.items():
    plt.plot(metric_df['Maturity'], metric_df['Absolute Error'], label=f'{model_name}_MAE')

plt.title('Comparison of MAE (sorted by Maturity)')
plt.xlabel('Maturity')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig(f'{figure_save_dir}/comparison_mae_maturity.png', dpi=300)
plt.show()

# 生成按Date分组的RMSE对比图
plt.figure(figsize=(10, 6))

for model_name, df in predictions.items():
    df['RMSE'] = (df['Real Closing Price'] - df['Predicted Closing Price']) ** 2
    date_rmse = df.groupby('Date')['RMSE'].mean().apply(np.sqrt)
    plt.plot(date_rmse.index, date_rmse.values, label=f'{model_name}_RMSE')

plt.title('Comparison of RMSE (sorted by Date)')
plt.xlabel('Date')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig(f'{figure_save_dir}/comparison_rmse_date.png', dpi=300)
plt.show()
