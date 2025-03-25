import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split  # 新增：数据划分
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")

# 数据准备
data = pd.read_csv("data", header=None, sep='\t', names=['参数', '结果'])
X = data[['参数']].values
y = data['结果'].values

# 1. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=50,  # 验证集大小为50
    random_state=42  # 固定随机种子以确保可复现
)

# 2. 标准化（仅基于训练集计算参数）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)  # 使用训练集的参数标准化验证集

# 3. 转换为张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# 4. 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证集无需 shuffle

# 模型配置
input_size = 1
hidden_size = 32
output_size = 1
depth = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义模型（与之前相同，但简化了设备处理）
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth):
        super(DNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.input_activation = nn.Tanh()
        self.hidden_activation = nn.ELU()
    
    def forward(self, x):
        out = self.input_activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.hidden_activation(layer(out))
        return self.output_layer(out)

model = DNN(input_size, hidden_size, output_size, depth).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环（新增验证阶段）
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(inputs)
    
    # 计算训练集平均损失
    avg_train_loss = train_loss / len(train_dataset)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * len(inputs)
    
    avg_val_loss = val_loss / len(val_dataset)
    
    # 输出结果
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

model.eval()

# 计算相对误差（基于训练集和验证集的总数据）
with torch.no_grad():
    # 训练集预测
    X_train_dev = X_train_tensor.to(device)
    y_train_pred = model(X_train_dev).cpu().numpy().flatten()
    y_train_true = y_train_tensor.cpu().numpy().flatten()
    
    # 验证集预测
    X_val_dev = X_val_tensor.to(device)
    y_val_pred = model(X_val_dev).cpu().numpy().flatten()
    y_val_true = y_val_tensor.cpu().numpy().flatten()
    
    # 合并所有预测和真实值
    y_pred_total = np.concatenate([y_train_pred, y_val_pred])
    y_true_total = np.concatenate([y_train_true, y_val_true])
    
    # 相对误差基于每个样本的真实值
    relative_error = np.abs(y_pred_total - y_true_total) / y_true_total

# 绘制相对误差曲线（与之前相同）
def relative_error_curve(relative_error, outputpath, title_fontsize=16, axis_title_fontsize=12):
    sorted_err = np.sort(relative_error)
    cumulative_prob = np.arange(1, len(sorted_err)+1) / len(sorted_err)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_err, cumulative_prob, label="Model", color='blue')
    plt.plot([0, max(sorted_err)], [0, 1], linestyle='--', color='gray', label="Perfect Prediction")
    plt.title("Cumulative Relative Error Curve", fontsize=title_fontsize)
    plt.xlabel("Relative Error", fontsize=axis_title_fontsize)
    plt.ylabel("Cumulative Probability", fontsize=axis_title_fontsize)
    plt.xlim(0, max(sorted_err) * 1.1)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(outputpath + "relative_error_curve.png", dpi=300)
    plt.show()

relative_error_curve(relative_error, outputpath="./")