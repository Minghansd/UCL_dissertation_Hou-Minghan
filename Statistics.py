import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# 加载数据
files = {
    'ANN': 'result/ANN_closing_price_predictions.xlsx',
    'BS': 'result/BS_model_theoretical_price_predictions.xlsx',
    'CNN': 'result/CNN_closing_price_predictions.xlsx',
    'Heston': 'result/Heston_model_theoretical_price_predictions.xlsx',
    'LSTM': 'result/LSTM_closing_price_predictions.xlsx'
}

# 读取数据并计算误差
errors = {}
for model, file_path in files.items():
    print(model)
    data = pd.read_excel(file_path)
    errors[model] = data['Real Closing Price'] - data['Predicted Closing Price']

# 计算 DM 统计量
def dm_test(e1, e2):
    diff = e1 - e2
    mean_diff = np.mean(diff)
    var_diff = np.var(diff, ddof=1)
    dm_stat = mean_diff / np.sqrt(var_diff / len(diff))
    return dm_stat

dm_results = pd.DataFrame(index=files.keys(), columns=files.keys())
for model1 in files.keys():
    for model2 in files.keys():
        if model1 != model2:
            dm_stat = dm_test(errors[model1], errors[model2])
            dm_results.loc[model1, model2] = round(dm_stat, 2)

# 打印 DM 统计量结果
print("DM Statistics:")
print(dm_results)

# 计算 WS 统计量
def loss_function(e):
    return np.mean(np.abs(e))

ws_results = pd.DataFrame(index=files.keys(), columns=files.keys())
for model1 in files.keys():
    for model2 in files.keys():
        if model1 != model2:
            losses = loss_function(errors[model1] - errors[model2])
            ws_stat, p_value = ttest_ind(errors[model1], errors[model2])
            ws_results.loc[model1, model2] = f'{p_value}'

# 打印 WS 统计量结果
print("WS Statistics:")
print(ws_results)

# 导出 DM 统计量结果到 Excel 文件
dm_results_file_path = 'Statistics/DM_Statistics.xlsx'
dm_results.to_excel(dm_results_file_path, index=True)
print(f"DM 统计量已导出到 {dm_results_file_path}")

# 导出 WS 统计量结果到 Excel 文件
ws_results_file_path = 'Statistics/WS_Statistics.xlsx'
ws_results.to_excel(ws_results_file_path, index=True)
print(f"WS 统计量已导出到 {ws_results_file_path}")