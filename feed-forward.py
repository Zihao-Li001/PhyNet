import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt  # 确保导入 matplotlib
sns.set_style("darkgrid")

# 数据准备
data = pd.read_csv("data", header=None, sep='\t', names=['参数', '结果'])
X = data[['参数']].values
y = data['结果'].values

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# 转换为张量（注意：此处应使用标准化后的 X_scaled）
X_tensor = torch.tensor(X, dtype=torch.float32)  # 修正：使用标准化后的数据
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型配置
input_size = 1
hidden_size = 32
output_size = 1
depth = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义模型
class DNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth, device='cpu'):
        super(DNN, self).__init__()
        
        # depth
        self.depth = depth
        
        # deploy layers
        self.input_layer = torch.nn.Linear(input_size, hidden_size).to(device)
        self.hidden_layers = nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size).to(device) for _ in range(depth)]) 
        self.output_layer = torch.nn.Linear(hidden_size, output_size).to(device)
        
        # setting activations 
        self.input_activation = torch.nn.Tanh()
        self.hidden_activation = torch.nn.ELU()
        self.output_activation = 'Linear'
    
    def forward(self, x):
        # input layer
        out = self.input_layer(x)
        out = self.input_activation(out)
        
        # hidden layers
        for h_layer in self.hidden_layers:
            out = h_layer(out)
            out = self.hidden_activation(out)

        # output layer
        out = self.output_layer(out)
        
        return out

# 初始化模型
model = DNN(input_size, hidden_size, output_size, depth, device).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 200
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 预测
new_params = torch.tensor([[0.2],[0.25], [4.5], [120.0]], dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    predictions = model(new_params)
    print("Predictions:", predictions.cpu().numpy())

# 计算相对误差
model.eval()
with torch.no_grad():
    # 使用标准化后的输入进行预测（关键修正点）
    predictions = model(X_tensor.to(device)).cpu().numpy().flatten()  
    targets = y_tensor.cpu().numpy().flatten()  # 原始目标值

    # 计算全局均值（如果数据没有分组）
    mean_value = np.mean(targets)
    relative_error = np.abs(predictions - targets) / mean_value  # 相对误差基于全局均值

# 绘制相对误差曲线（修正函数）
def relative_error_curve(relative_error, outputpath, title_fontsize=16, axis_title_fontsize=12):
    sorted_err = np.sort(relative_error)
    cumulative_prob = np.arange(len(sorted_err)) / len(sorted_err)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_err, cumulative_prob, label="Model", color='blue')
    plt.plot([0, max(sorted_err)], [0, 1], linestyle='--', color='gray', label="Perfect Prediction")
    plt.title("Cumulative Relative Error Curve", fontsize=title_fontsize)
    plt.xlabel("Relative Error", fontsize=axis_title_fontsize)
    plt.ylabel("Cumulative Probability", fontsize=axis_title_fontsize)
    plt.xlim(0, max(sorted_err) * 1.1)  # 自适应坐标轴范围
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(outputpath + "relative_error_curve.png", dpi=300)
    plt.show()

relative_error_curve(relative_error, outputpath="./")