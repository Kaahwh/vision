import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from handwriting.pytorch.model import *


torch.manual_seed(89)

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载MNIST数据集在当前目录下的dataset文件夹下
train_dataset = torchvision.datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./dataset', train=False, transform=transform, download=True)

# 创建数据加载器
# 训练集数据打乱
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# 测试集数据不打乱
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
nw = NET()
loss_fun = nn.CrossEntropyLoss()
optim = optim.SGD(nw.parameters(), lr=0.01)

# 用于可视化
writer = SummaryWriter("logs")


# 开始训练十轮
total_train_step = 0 # 记录训练步数
for epoch in range(10):  # 训练10个周期
    print("第{}轮开始".format(epoch + 1))
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        print(inputs.shape)
        # 训练步骤
        optim.zero_grad()
        outputs = nw(inputs)
        loss = loss_fun(outputs, labels)
        loss.backward()
        optim.step()
        running_loss += loss.item()
        writer.add_scalar("train", loss, total_train_step)
        total_train_step = total_train_step + 1
    print(f'第{epoch+1}轮Loss: {running_loss / len(train_loader)}')

print('训练完成')

# 在测试集上测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = nw(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
# 模型保存
torch.save(nw, "net.pth")

# 输出准确率
print(f'测试集准确率: {100 * correct / total}%')



