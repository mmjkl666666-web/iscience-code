import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score

# 1. 数据加载与预处理
data = pd.read_csv('D:/Desktop/机器学习/train/your_data1.csv')

# 定义特征和目标变量
features = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol', 'sgnumber',
            'Z', 'Preparation temperature', 'concentration', 'frequency', 'temperature']
X = data[features]
y = data[['dielectric constant', 'dielectric loss']].values

# 2. 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['sgnumber']),
        ('num', StandardScaler(), [f for f in features if f != 'sgnumber'])
    ]
)

# 数据预处理
X_processed = preprocessor.fit_transform(X)
if hasattr(X_processed, 'toarray'):
    X_processed = X_processed.toarray()

# 3. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_processed.astype(np.float32),
    y.astype(np.float32),
    test_size=0.2,
    random_state=42
)


# 4. 定义物理约束函数（示例：介电常数随体积变化的导数约束）
def apply_physics_constraints(inputs, outputs):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        # 假设模型预测介电常数（输出第一列）和介电损耗（输出第二列）
        pred_const = outputs[:, 0]

    # 计算介电常数对体积的导数（假设vol是预处理后的第6列）
    dv_dvol = tape.gradient(pred_const, inputs)[:, 6]

    # 物理约束示例：介电常数随体积增大而减小（可根据实际物理规律修改）
    physics_loss = tf.reduce_mean(tf.square(tf.nn.relu(dv_dvol)))  # 惩罚正导数

    return physics_loss


# 5. 构建PINN模型
class PhysicsInformedNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='tanh'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(2)  # 同时输出介电常数和介电损耗
        ])

    def call(self, inputs):
        return self.dense_layers(inputs)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # 前向传播
            y_pred = self(x, training=True)

            # 数据拟合损失
            mse_loss = tf.reduce_mean(tf.square(y - y_pred))

            # 物理约束损失
            physics_loss = apply_physics_constraints(x, y_pred)

            # 总损失（调整alpha参数平衡损失项）
            total_loss = mse_loss + 0.3 * physics_loss

        # 反向传播
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'loss': total_loss, 'mse_loss': mse_loss, 'physics_loss': physics_loss}


# 6. 模型配置
model = PhysicsInformedNN()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# 7. 模型训练
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# 8. 预测与评估
y_pred = model.predict(X_test)
r2_const = r2_score(y_test[:, 0], y_pred[:, 0])
r2_loss = r2_score(y_test[:, 1], y_pred[:, 1])

# 9. 可视化结果
plt.figure(figsize=(14, 6), dpi=100)

# 介电常数预测
plt.subplot(1, 2, 1)
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.6, c='#1f77b4', edgecolor='w', s=80)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Dielectric Constant', fontsize=12)
plt.ylabel('Predicted Dielectric Constant', fontsize=12)
plt.title(f'Dielectric Constant (R² = {r2_const:.3f})')
plt.grid(True, alpha=0.3, linestyle=':')

# 介电损耗预测
plt.subplot(1, 2, 2)
plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.6, c='#ff7f0e', edgecolor='w', s=80)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Dielectric Loss', fontsize=12)
plt.ylabel('Predicted Dielectric Loss', fontsize=12)
plt.title(f'Dielectric Loss (R² = {r2_loss:.3f})')
plt.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.show()