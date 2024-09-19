import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取原始数据
file_path = 'data/BS.xlsx'  # 将其替换为您的实际文件路径
original_data = pd.read_excel(file_path)
# print(original_data)

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
for model_name, file_path_ in files.items():
    predictions[model_name] = pd.read_excel(file_path_)
    # print(pd.read_excel(file_path))

# 合并`距离到期日天数`与`报价时间`到预测结果
for model_name, df in predictions.items():
    if model_name in ['BS', 'Heston']:
        df['Maturity'] = original_data['距离到期日天数']
        df['Date'] = original_data['报价时间']

    else:
        df['Maturity'] = original_data['距离到期日天数'][-len(df):].values
        df['Date'] = original_data['报价时间'][-len(df):].values
    # 转换日期列为datetime格式，并过滤时间范围
    df['Date'] = pd.to_datetime(df['Date'])
    # print(df['Date'])

    if 'O510050ivf_20180101至20181231.xlsm' in file_path:
        df = df[(df['Date'] >= '2018-07-01') & (df['Date'] <= '2018-12-31')]
        predictions[model_name] = df
        print(12)
    else:
        df = df[(df['Date'] >= '2020-07-01') & (df['Date'] <= '2020-12-31')]
        predictions[model_name] = df

# 计算每个模型的MAE和RMSE并根据日期进行分组
mae_by_date = {}
rmse_by_date = {}
for model_name, df in predictions.items():
    df['Error'] = df['Real Closing Price'] - df['Predicted Closing Price']
    df['Absolute Error'] = df['Error'].abs()
    df['Squared Error'] = df['Error'] ** 2
    mae_by_date[model_name] = df.groupby('Date')['Absolute Error'].mean()
    rmse_by_date[model_name] = df.groupby('Date')['Squared Error'].mean().apply(np.sqrt)

# 生成按日期分组的MAE对比图
plt.figure(figsize=(10, 6))

for model_name, mae_df in mae_by_date.items():
    plt.plot(mae_df.index, mae_df.values, label=f'{model_name}_MAE')

plt.title('Comparison of MAE (sorted by Date, sliding)')
plt.xlabel('Date')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig(f'{figure_save_dir}/rubust_mae_date_filtered.png', dpi=300)
plt.show()

# 生成按日期分组的RMSE对比图
plt.figure(figsize=(10, 6))

for model_name, rmse_df in rmse_by_date.items():
    plt.plot(rmse_df.index, rmse_df.values, label=f'{model_name}_RMSE')

plt.title('Comparison of RMSE (sorted by Date, sliding)')
plt.xlabel('Date')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig(f'{figure_save_dir}/rubust_rmse_date_filtered.png', dpi=300)
plt.show()
