import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

# 1. 数据加载
data = pd.read_csv('D:/Desktop/机器学习/train/your_data1.csv')

# 2. 数据预处理
X = data[['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol', 'sgnumber',
          'Z', 'Preparation temperature', 'concentration', 'frequency', 'temperature']]
y = data[['dielectric constant', 'dielectric loss']]

# 3. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['sgnumber']),
        ('num', StandardScaler(), ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol',
                                   'Z', 'Preparation temperature', 'concentration',
                                   'frequency', 'temperature'])
    ]
)

# 应用预处理
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)


# 5. 构建神经网络模型
def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,),
              kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(64, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(2)  # 同时输出两个目标
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )
    return model


# 6. 初始化模型
model = build_model(X_train_proc.shape[1])

# 7. 训练配置
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# 8. 模型训练
history = model.fit(
    X_train_proc, y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0  # 关闭训练日志输出
)

# 9. 预测与评估
y_pred = model.predict(X_test_proc)

# 10. 可视化结果
plt.figure(figsize=(12, 5))

# 介电常数散点图
plt.subplot(1, 2, 1)
plt.scatter(y_test['dielectric constant'], y_pred[:, 0],
            alpha=0.6, color='#1f77b4', edgecolor='w')
plt.plot([y_test.min()[0], y_test.max()[0]],
         [y_test.min()[0], y_test.max()[0]], 'r--')
plt.xlabel('Actual Dielectric Constant', fontsize=12)
plt.ylabel('Predicted Dielectric Constant', fontsize=12)
plt.title(f'Dielectric Constant (R²={r2_score(y_test.iloc[:, 0], y_pred[:, 0]):.3f})')
plt.grid(True, alpha=0.3)

# 介电损耗散点图
plt.subplot(1, 2, 2)
plt.scatter(y_test['dielectric loss'], y_pred[:, 1],
            alpha=0.6, color='#ff7f0e', edgecolor='w')
plt.plot([y_test.min()[1], y_test.max()[1]],
         [y_test.min()[1], y_test.max()[1]], 'r--')
plt.xlabel('Actual Dielectric Loss', fontsize=12)
plt.ylabel('Predicted Dielectric Loss', fontsize=12)
plt.title(f'Dielectric Loss (R²={r2_score(y_test.iloc[:, 1], y_pred[:, 1]):.3f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()