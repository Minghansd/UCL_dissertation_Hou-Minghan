import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from alibi.explainers import ALE

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子
tf.random.set_seed(42)
random.seed(42)

# 读取数据
file_path = 'data/BS.xlsx'
model_name = 'LSTM'
target = '理论价格'
data = pd.read_excel(file_path, index_col=0)
data.fillna(method='ffill', inplace=True)

# 查看列名，确保选择的列正确
print(data.columns)

# 特征列名（中文）
feature = [
    "执行价",
    "开盘价",
    "最高价",
    "最低价",
    "收盘价",
    "成交量",
    "持仓量",
    "成交额",
    "标的收盘价",
]

# 特征列名（英文）
feature_names = {
    "执行价": "Strike Price",
    "开盘价": "Opening Price",
    "最高价": "Highest Price",
    "最低价": "Lowest Price",
    "收盘价": "Closing Price",
    "成交量": "Volume",
    "持仓量": "Open Interest",
    "成交额": "Turnover",
    "标的收盘价": "Underlying Closing Price",
}

# 生成三阶滞后特征
lag = 3
for col in feature:
    data[f'{feature_names[col]}_lag_{lag}'] = data[col].shift(lag)

# 丢弃有NaN值的行，因为滞后特征会引入NaN
data.dropna(inplace=True)

# 选择三阶滞后特征列（英文名称）
lagged_features = [f'{feature_names[col]}_lag_{lag}' for col in feature]

# 提取滞后特征作为输入特征
X = data[lagged_features].values
y = data[target].values.reshape(-1, 1)

# 获取特征名（英文）
features = lagged_features

# 数据归一化
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 构建时间滑窗数据
def create_dataset(X, y, window_size=30):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)

window_size = 30
X_windowed, y_windowed = create_dataset(X_scaled, y_scaled)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_windowed, y_windowed, test_size=0.3, random_state=42, shuffle=False)

# 定义手动网格搜索的超参数
units_options = [50, 100]
dropout_options = [0.2, 0.3]
batch_size_options = [1024, 2048]
epochs_options = [10, 20]

# 用于记录最好的模型参数和结果
best_params = None
best_rmse = float('inf')

# 手动网格搜索
for units in units_options:
    for dropout_rate in dropout_options:
        for batch_size in batch_size_options:
            for epochs in epochs_options:
                print(f"Training with units={units}, dropout_rate={dropout_rate}, batch_size={batch_size}, epochs={epochs}")

                # 构建LSTM模型
                model = Sequential()
                model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                model.add(Dropout(dropout_rate))
                model.add(LSTM(units=units, return_sequences=False))
                model.add(Dropout(dropout_rate))
                model.add(Dense(units=1))

                # 编译模型
                model.compile(optimizer='adam', loss='mean_squared_error')

                # 训练模型
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                    validation_data=(X_test, y_test), verbose=1)

                # 预测
                y_pred_scaled = model.predict(X_test)

                # 数据反归一化
                y_test_unscaled = scaler_y.inverse_transform(y_test)
                y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled)

                # 计算评价指标
                rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))

                print(f"RMSE: {rmse}")

                # 更新最佳参数和结果
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {
                        'units': units,
                        'dropout_rate': dropout_rate,
                        'batch_size': batch_size,
                        'epochs': epochs
                    }

print(f"Best RMSE: {best_rmse} with parameters {best_params}")

# 使用最佳参数重新训练模型
units = best_params['units']
dropout_rate = best_params['dropout_rate']
batch_size = best_params['batch_size']
epochs = best_params['epochs']

# 构建最佳LSTM模型
model = Sequential()
model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(dropout_rate))
model.add(LSTM(units=units, return_sequences=False))
model.add(Dropout(dropout_rate))
model.add(Dense(units=1))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练最佳模型
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

# 预测
y_pred_scaled = model.predict(X_test)

# 数据反归一化
y_test = scaler_y.inverse_transform(y_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 计算最终评价指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Final RMSE: {rmse}')
print(f'Final MAE: {mae}')
print(f'Final R²: {r2}')

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

# 将真实值和预测值导出到Excel文件
result_file_path = f'{result_save_dir}/{model_name}_closing_price_predictions.xlsx'
result_df = pd.DataFrame({
    'Real Closing Price': y_test.flatten(),
    'Predicted Closing Price': y_pred.flatten()
})
result_df.to_excel(result_file_path, index=False)
print(f'Results saved to {result_file_path}')

# 绘制真实值与预测值对比图
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(y_test, color='blue', label='Real Closing Price')
plt.plot(y_pred, color='red', label='Predicted Closing Price')
plt.title('Closing Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig(f'{figure_save_dir}/{model_name}_Closing_Price_Prediction.png', dpi=300, bbox_inches='tight')
plt.show()

# 定义一个适应LSTM模型的预测函数
def lstm_predict(X):
    X_reshaped = X.reshape((X.shape[0], window_size, X_test.shape[2]))  # 调整输入的维度
    return model.predict(X_reshaped)

# 展平滑窗数据以便ALE解释
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# 计算和绘制ALE图
ale_explainer = ALE(lstm_predict, feature_names=features)
ale_exp = ale_explainer.explain(X_test_flattened)

# 创建一个2x3的子图布局
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 绘制每个特征的ALE图
for i, ax in enumerate(axs.flatten()[:6]):
    ax.plot(ale_exp.feature_values[i], ale_exp.ale_values[i], label='ALE')
    ax.set_title(features[i])
    ax.set_xlabel('Variable Value')
    ax.set_ylabel('ALE Value')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(f'{figure_save_dir}/{model_name}_ALE_Plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 计算ALE累积范围作为特征重要性
ale_range = {}
for i, feature in enumerate(features):
    ale_range[feature] = ale_exp.ale_values[i].max() - ale_exp.ale_values[i].min()

# 将特征按重要性排序
sorted_features = sorted(ale_range.items(), key=lambda x: x[1], reverse=True)
sorted_feature_names, sorted_ale_values = zip(*sorted_features)

# 绘制特征重要性排序图
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, sorted_ale_values, color='skyblue')
plt.xlabel('ALE Range')
plt.ylabel('Feature Names')
plt.title(f'{model_name} ALE Range All')
plt.gca().invert_yaxis()  # 使得最高的重要性在上方
plt.grid(True)

plt.savefig(f'{figure_save_dir}/{model_name}_ALE_Range.png', dpi=300, bbox_inches='tight')
plt.show()
