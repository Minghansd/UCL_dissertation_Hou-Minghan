import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# 读取上传的文件
file_path = 'data/BS.xlsx'  # 将其替换为你的实际文件路径
data = pd.read_excel(file_path)

# 根据文件名确定保存目录
if 'O510050ivf_20180101至20181231.xlsm' in file_path:
    result_save_dir = 'result1'
    describe_save_dir = 'describe1'
else:
    result_save_dir = 'result2'
    describe_save_dir = 'describe2'

# 如果保存目录不存在，则创建
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)
if not os.path.exists(describe_save_dir):
    os.makedirs(describe_save_dir)

# 计算Moneyness
data['Moneyness'] = data['标的收盘价'] / data['执行价']

# 对数据进行分组
bins = [0, 0.97, 1.03, float('inf')]
labels = ['lt_0.97', '0.97-1.03', 'gte_1.03']
data['Moneyness_Group'] = pd.cut(data['Moneyness'], bins=bins, labels=labels)

# 中文列名到英文列名的映射
column_mapping = {
    '无风险利率': 'Risk-Free Rate',
    '标的收盘价': 'Underlying Closing Price',
    '经插值的隐含波动率': 'Interpolated Implied Volatility',
    '执行价': 'Strike Price',
    '距离到期日时间（折算成年）': 'Time to Maturity (Years)',
    '理论价格': 'Theoretical Price'
}

# 计算每个分组的统计信息并导出到Excel文件
for group in labels:
    print(f"--- Moneyness Group: {group} ---")
    group_data = data[data['Moneyness_Group'] == group]
    description = group_data[
        ['无风险利率', '标的收盘价', '经插值的隐含波动率', '执行价', '距离到期日时间（折算成年）', '理论价格']].describe()

    # 映射列名为英文
    description.rename(columns=column_mapping, inplace=True)
    print(description)
    print("\n")

    # 导出到Excel文件
    output_file_path = f'{describe_save_dir}/moneyness_statistics_{group}.xlsx'
    description.to_excel(output_file_path)
    print(f"统计表已保存到 {output_file_path}")

# 继续处理预测结果
# 读取原始数据文件
original_data_path = 'data/BS.xlsx'
original_data = pd.read_excel(original_data_path)

# 获取 "距离到期日天数" 列
maturity_days = original_data['距离到期日天数'].values

# 读取BS模型的预测结果
bs_predictions_path = f'{result_save_dir}/BS_model_theoretical_price_predictions.xlsx'
bs_predictions = pd.read_excel(bs_predictions_path)

# 将 "距离到期日天数" 添加到BS模型的预测结果
bs_predictions['Days to Maturity'] = maturity_days[:len(bs_predictions)]
bs_predictions.rename(columns={'距离到期日天数': 'Days to Maturity'}, inplace=True)
bs_predictions.to_excel(f'{result_save_dir}/BS_model_with_maturity.xlsx', index=False)

# 处理Heston模型
heston_predictions_path = f'{result_save_dir}/Heston_model_theoretical_price_predictions.xlsx'
heston_predictions = pd.read_excel(heston_predictions_path)

# 将 "距离到期日天数" 添加到Heston模型的预测结果
heston_predictions['Days to Maturity'] = maturity_days[:len(heston_predictions)]
heston_predictions.rename(columns={'距离到期日天数': 'Days to Maturity'}, inplace=True)
heston_predictions.to_excel(f'{result_save_dir}/Heston_model_with_maturity.xlsx', index=False)

# 获取划分数据集后的训练集和测试集的长度
train_size = int(len(original_data) * 0.7)
test_size = len(original_data) - train_size

# 处理ANN模型
ann_predictions_path = f'{result_save_dir}/ANN_closing_price_predictions.xlsx'
ann_predictions = pd.read_excel(ann_predictions_path)

# 取出测试集对应的 "距离到期日天数"
ann_maturity_days = maturity_days[train_size:train_size + len(ann_predictions)]
ann_predictions['Days to Maturity'] = ann_maturity_days
ann_predictions.rename(columns={'距离到期日天数': 'Days to Maturity'}, inplace=True)
ann_predictions.to_excel(f'{result_save_dir}/ANN_model_with_maturity.xlsx', index=False)

# 处理CNN模型
cnn_predictions_path = f'{result_save_dir}/CNN_closing_price_predictions.xlsx'
cnn_predictions = pd.read_excel(cnn_predictions_path)

# 取出测试集对应的 "距离到期日天数"
cnn_maturity_days = maturity_days[train_size:train_size + len(cnn_predictions)]
cnn_predictions['Days to Maturity'] = cnn_maturity_days
cnn_predictions.rename(columns={'距离到期日天数': 'Days to Maturity'}, inplace=True)
cnn_predictions.to_excel(f'{result_save_dir}/CNN_model_with_maturity.xlsx', index=False)

# 处理LSTM模型
lstm_predictions_path = f'{result_save_dir}/LSTM_closing_price_predictions.xlsx'
lstm_predictions = pd.read_excel(lstm_predictions_path)

# 取出测试集对应的 "距离到期日天数"
lstm_maturity_days = maturity_days[train_size:train_size + len(lstm_predictions)]
lstm_predictions['Days to Maturity'] = lstm_maturity_days
lstm_predictions.rename(columns={'距离到期日天数': 'Days to Maturity'}, inplace=True)
lstm_predictions.to_excel(f'{result_save_dir}/LSTM_model_with_maturity.xlsx', index=False)

# 读取预测数据
models = ['BS', 'Heston', 'ANN', 'LSTM', 'CNN']
files = {
    'ANN': f'{result_save_dir}/ANN_model_with_maturity.xlsx',
    'BS': f'{result_save_dir}/BS_model_with_maturity.xlsx',
    'CNN': f'{result_save_dir}/CNN_model_with_maturity.xlsx',
    'Heston': f'{result_save_dir}/Heston_model_with_maturity.xlsx',
    'LSTM': f'{result_save_dir}/LSTM_model_with_maturity.xlsx'
}

# 读取数据
data = {}
for model in models:
    data[model] = pd.read_excel(files[model])

# 计算Moneyness
for model in models:
    print(model)
    data[model]['Moneyness'] = data[model]['Real Closing Price'] / data[model]['Predicted Closing Price']

# 分组区间
moneyness_bins = [0, 0.97, 1.03, float('inf')]
moneyness_labels = ['<0.97', '0.97 ~ 1.03', '≥1.03']

maturity_bins = [1, 30, 60, 90, 120, 150, 180, 210, 240]
maturity_labels = ['1-30', '31-60', '61-90', '91-120', '121-150', '151-180', '181-210', '211-240']

# 创建一个空的 DataFrame 来存储RMSE结果
rmse_table = pd.DataFrame(columns=['Moneyness', 'Maturity'] + models)

# 计算RMSE并存储到表格中
for m_label in moneyness_labels:
    for t_label in maturity_labels:
        rmse_row = {'Moneyness': m_label, 'Maturity': t_label}
        for model in models:
            model_data = data[model]
            model_data['Moneyness_Group'] = pd.cut(model_data['Moneyness'], bins=moneyness_bins, labels=moneyness_labels)
            model_data['Maturity_Group'] = pd.cut(model_data['Days to Maturity'], bins=maturity_bins, labels=maturity_labels)

            # 筛选当前分组下的数据
            group_data = model_data[(model_data['Moneyness_Group'] == m_label) & (model_data['Maturity_Group'] == t_label)]

            if len(group_data) > 0:
                rmse_value = np.sqrt(mean_squared_error(group_data['Real Closing Price'], group_data['Predicted Closing Price']))
            else:
                rmse_value = np.nan  # 如果该分组下没有数据，则设为NaN

            rmse_row[model] = rmse_value

        rmse_table = rmse_table.append(rmse_row, ignore_index=True)

# 打印结果
print(rmse_table)

# 保存结果到文件
rmse_table.to_excel(f'{result_save_dir}/RMSE_Table.xlsx', index=False)
