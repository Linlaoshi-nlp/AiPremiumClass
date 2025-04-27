import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

in_features = 28 * 28
batch_size = 128  # eg: 64

# 1. 数据加载与预处理
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
    ]
)

# 加载 MNIST 数据集
train_dataset = datasets.KMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.KMNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

len_test = len(test_dataset)
len_train = len(train_dataset)
sum_batch = math.ceil(len_train / batch_size)
print(f"训练集大小: {len_train}, 测试集大小: {len_test}", "批次数量:", sum_batch)
for images, labels in train_loader:
    print(images.shape)  # 输出图像的形状
    print(labels.shape)  # 输出标签的形状
    print(labels)  # 输出标签
    print(images.reshape(-1, in_features).shape)  # 输出展平后的图像形状 (128, 784)
    break


# 2. 定义 CNN 模型 - 假设输入的图像是单通道的灰度图像，其形状为 (N, 1, 28, 28)，这里的 N 表示批量大小
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层：输入1通道，输出32通道，卷积核大小3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 定义卷积层：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入大小 = 特征图大小 * 通道数
        self.fc2 = nn.Linear(128, 10)  # 10 个类别

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = F.max_pool2d(x, 2)  # 最大池化
        x = F.relu(self.conv2(x))  # 第二层卷积 + ReLU
        x = F.max_pool2d(x, 2)  # 最大池化
        x = x.view(-1, 64 * 7 * 7)  # 展平操作
        x = F.relu(self.fc1(x))  # 全连接层 + ReLU
        x = self.fc2(x)  # 全连接层输出
        return x


# 创建模型实例
model = SimpleCNN()

# 直接使用 nn.Sequential 定义模型
# model = nn.Sequential(
#     nn.Linear(in_features, 256),  # in_features = 28 * 28
#     nn.ReLU(),  # 激活函数
#     nn.Linear(256, 10),  # 隐藏层
# )


# 辅助函数：展平数据
def compose_data(data):
    global model, in_features
    if type(model) == nn.Sequential:
        return data.reshape(-1, in_features)
    else:
        return data


# 3. 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器


# 4. 训练模型
def train(model, train_loader, optimizer, loss_function, epochs=5):
    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # 清空梯度
            output = model(compose_data(data))  # 前向传播
            loss = loss_function(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()  # 累加损失

            if batch_idx % 100 == 0:  # 每100个批次输出一次损失
                print(
                    f"Epoch: {epoch} [{batch_idx * len(data)}/{len_train}] "
                    f"Loss: {loss.item():.6f} "
                    f"Loss_Total: {round(total_loss, 2)} "
                )
        print(f"Train Epoch [{epoch+1}/{epochs}], Loss: {total_loss / sum_batch:.4f}")


# 5. 测试模型
def verify(model, test_loader):
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            output = model(compose_data(data))  # 前向传播

            test_loss += loss_function(output, target).item()  # 累加损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            pred_values = pred.reshape(-1)

            counts = torch.eq(target, pred_values)
            print(pred_values[:8])
            print(target[:8])
            print(output.shape)
            print("----------------", counts.sum())
            correct += counts.sum().item()  # 统计正确预测的数量

    test_loss /= len(test_loader.dataset)  # 平均损失
    # 输出测试结果
    print(
        f"Test set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len_test} ({100.0 * correct / len_test:.2f}%)"
    )


train(model, train_loader, optimizer, loss_function, epochs=8)
verify(model, test_loader)

# 6. 保存模型
torch.save(model.state_dict(), "models/kmnist_cnn.pth")  # 保存模型参数

# 7. 加载模型
model_data = torch.load("models/kmnist_cnn.pth")  # 加载模型参数
model.load_state_dict(model_data)  # 将参数加载到模型中

# 8. 测试加载的模型
verify(model, test_loader)


# 9. 可视化测试结果
def draw_predictions(test_loader, model):
    dataIter = iter(test_loader)
    images, labels = next(dataIter)
    outputs = model(compose_data(images))  # 前向传播
    predictions = torch.argmax(outputs, 1)
    print("outputs values:", outputs.shape)
    print("True labels:", labels[:10])
    print("Predicted labels:", predictions[:10])

    fig, axes = plt.subplots(1, 10, figsize=(12, 4))
    for i in range(10):
        axes[i].imshow(images[i][0], cmap="gray")
        axes[i].set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
        axes[i].axis("off")
    plt.show()


draw_predictions(test_loader, model)
