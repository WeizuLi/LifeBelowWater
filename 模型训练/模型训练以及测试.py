import matplotlib
import matplotlib.pyplot as plt
import time
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torchvision import models
import torch.optim as optim
from ourModel import LwzModel

matplotlib.rc("font", family='SimHei')  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
# 训练集图像预处理
train_transform = transforms.Compose([
    # 缩放
    transforms.Resize(256),
    # 裁剪图片
    transforms.RandomCrop(224),
    # transforms.CenterCrop(224),
    # 图像增强
    # transforms.RandomRotation()
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # 图像归一化，ToTensor()能够把灰度范围从0-255变换到0-1之间，transform.Normalize()则把0-1变换到(-1,1)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])
# 测试集图像预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    # 裁剪图片
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 载入数据集
dataset_path = '../海洋生物图片采集/LifeBelowWater_dataset'
train_set_path = os.path.join(dataset_path, 'train')
test_set_path = os.path.join(dataset_path, 'test')
# 按照图片预处理方式载入数据集
train_set = datasets.ImageFolder(train_set_path, train_transform)
test_set = datasets.ImageFolder(test_set_path, test_transform)
# print(train_set)
print('训练集图像数量', len(train_set))
print('类别个数', len(train_set.classes))
print('各类别名称', train_set.classes)
print('测试集图像数量', len(test_set))
print('类别个数', len(test_set.classes))
print('各类别名称', test_set.classes)
# 讲类别名称保存，并且和索引号一一对应
class_names = train_set.classes
# 生成  类别：索引号 的字典，再反转，方便之后按照索引查询类别
dict = train_set.class_to_idx
index_to_class = {}
for i, ele in enumerate(dict, 0):
    index_to_class[i] = ele
print(index_to_class)
np.save('index_to_classes.npy', index_to_class)
np.save('class_to_idx.npy', dict)

# 数据加载器
batch_size = 32
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels)

# 经过数据集加载器处理过的图像
# idx = 15
# # 转为(224, 224, 3)
# plt.imshow(images[idx].detach().cpu().squeeze().numpy().transpose((1,2,0)))
# label = labels[idx].item()
# pred_classname = index_to_class[label]
# plt.title('label:'+pred_classname)
# plt.show()
# # 原始图像
# idx = 15
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# plt.imshow(np.clip(images[idx].detach().cpu().squeeze().numpy().transpose((1,2,0)) * std + mean, 0, 1))
# plt.title('label:'+ pred_classname)
# plt.show()

# 采用自编写模型训练
# model= LwzModel.LwzModule()
# model=torch.load('model_weight/Densenet121False_batch32_epoch30_33c_20221118.pth')
# 载入模型进行训练，采用resnet18网络模型 ---------
# model = models.resnet18(pretrained=False)  # 载入模型结构，
# model = models.resnet50(pretrained=False)
model = models.densenet121(pretrained=False)  # 载入模型结构，载入预训练权重参数
print(model)
# # model = models.resnet18(pretrained=False)  # 只载入模型结构，不载入预训练权重参数
# # 模型最后一层的Linear的输出维度，改为 类型的个数
# model.fc = nn.Linear(model.fc.in_features, len(train_set.classes))
model.classifier = nn.Linear(model.classifier.in_features, len(train_set.classes))
# # 优化器 只微调最后一层权重
# optimizer = optim.Adam(model.fc.parameters())
# # 调整全体权重
optimizer = optim.Adam(model.parameters())
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)
# # 损失函数采用交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
model = model.to(device)

# model = torch.load('model_weight/resnet18True_batch32_epoch30_33c.pth')
# model = model.eval().to(device)

# 训练模型
train_Accuracy=[]
train_loss=[]
def train(epoch):
    model.train()
    running_loss = 0
    train_total = 0
    train_correct = 0
    for batch_index, data in enumerate(train_loader, 0):
        inputs, labels = data
        # 数据放入显卡
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)
        train_total+=labels.size(0)
        running_loss += loss.item()
        train_correct += (predicted == labels).sum().item()
        if batch_index==124:
            train_loss.append(running_loss/124)
        if batch_index % 32 == 31:
            print(f"[{epoch},{batch_index},loss={running_loss / 32}]")
            with open('output/train_loss_log.csv', 'a+', encoding='utf-8') as f:
                f.write(f"[{epoch},{batch_index},loss={running_loss / 32}]\n")
    print(f"[{epoch},准确率={100*(train_correct / train_total)}%,{train_correct}/{train_total}]")
    train_Accuracy.append(train_correct/train_total)
    with open('output/train_loss_log.csv', 'a+', encoding='utf-8') as f:
        f.write(f"[{epoch},准确率={100*(train_correct / train_total)}%,{train_correct}/{train_total}]")


# 测试模型
Accuracy = []


def test():
    model.eval()
    total = 0
    correct = 0
    running_loss=0
    with torch.no_grad():
        for batch_index, data in enumerate(test_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _,predicted = torch.max(outputs.data, dim=1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            if batch_index % 32 == 31:
                print(f"[{epoch},{batch_index},loss={running_loss / 32}]")
    Accuracy.append(correct / total)
    print(f"准确率:{100 * correct / total}%，{correct}/{total}")
    with open('output/test_Accuracy_log.csv', 'a+', encoding='utf-8') as f:
        f.write(f"准确率:{100 * correct / total}%，{correct}/{total}\n")


if __name__ == '__main__':
    i = 0
    for epoch in range(1, 31):
        train(epoch)
        test()
        i = epoch
    epoch_list = list(range(1, i + 1))
    # plt.xlim(0,32)
    # plt.ylim(0.0, 1.0)
    # plt.plot(epoch_list, Accuracy)
    # plt.plot(epoch_list,train_Accuracy)
    # plt.xlabel("epoch")
    # plt.ylabel("Accuracy")
    # plt.ylabel("trainAccuracy")
    # plt.show()
    # plt.plot(epoch_list, Accuracy)
    # plt.xlabel("epoch")
    # plt.ylabel("Accuracy")
    # plt.show()
    # plt.plot(epoch_list,train_Accuracy)
    # plt.xlabel("epoch")
    # plt.ylabel("trainAccuracy")
    # plt.show()
    plt.xlim(0,32)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(epoch_list, train_loss)
    plt.show()

    # 保存模型
    # torch.save(model, 'model_weight/LwzModel18False_batch32_epoch30_33c_202211.pth')
