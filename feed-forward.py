import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")

# 数据准备
data = pd.read_csv("data", header=None, sep='\t', names=['参数', '结果'])
X = data[['参数']].values
y = data['结果'].values

# 数据划分
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=40,
    random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
y_train_scaled = scaler.fit_transform(y_train.reshape(-1,1))
y_val_scaled = scaler.transform(y_val.reshape(-1,1))

# 转换为张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

# # 数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# # 模型定义（与之前相同）
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth):
        super(DNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.input_activation = nn.Tanh()
        self.hidden_activation = nn.ELU()
        self.xxx = input_size
    
    def forward(self, x):
        out = self.input_activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.hidden_activation(layer(out))
        return self.output_layer(out)

model = DNN(input_size=1, hidden_size=32, output_size=1, depth=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # 初始化损失记录列表
train_loss_history = []
val_loss_history = []

num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs, targets = batch[0].to(device), batch[1].to(device)
        print("\n=======model======\n ",model)
        print("\n=======inputs=======\n",inputs.device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
#     avg_train_loss = train_loss / len(train_dataset)
    
#     # 验证阶段
#     model.eval()
#     val_loss = 0.0
#     y_val_true_epoch = []
#     y_val_pred_epoch = []
#     with torch.no_grad():
#         for batch in val_loader:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             model.to(device)
#             inputs, targets = batch[0].to(device), batch[1].to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             val_loss += loss.item() * len(inputs)
#             y_val_true_epoch.extend(targets.cpu().numpy().flatten())
#             y_val_pred_epoch.extend(outputs.cpu().numpy().flatten())
    
#     avg_val_loss = val_loss / len(val_dataset)
    
#     # 记录损失
#     train_loss_history.append(avg_train_loss)
#     val_loss_history.append(avg_val_loss)
    
#     if (epoch+1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{num_epochs}]")
#         print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# # 绘制损失曲线
# plt.figure(figsize=(12, 6))
# plt.plot(train_loss_history, label='Training Loss')
# plt.plot(val_loss_history, label='Validation Loss')
# plt.title('Training and Validation Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.savefig("loss_curve.png", dpi=300)
# plt.show()

# # 绘制验证集预测与真实值散点图
# y_val_true = np.array(y_val_true_epoch)
# y_val_pred = np.array(y_val_pred_epoch)
# plt.figure(figsize=(10, 6))
# plt.scatter(y_val_true, y_val_pred, alpha=0.7)
# plt.plot([min(y_val_true), max(y_val_true)], [min(y_val_true), max(y_val_true)], 
#          '--', color='red', label='Perfect Prediction')
# plt.title('Validation: True vs Predicted Values')
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.legend()
# plt.grid(True)
# plt.savefig("validation_scatter.png", dpi=300)
# plt.show()