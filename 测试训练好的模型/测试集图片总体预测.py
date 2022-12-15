import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from torchvision import datasets
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

test_dataset_dir='../海洋生物图片采集/LifeBelowWater_dataset/test'
# dataloader
test_dataset_loader=datasets.ImageFolder(test_dataset_dir,test_transform)
index_to_classes=np.load('../模型训练/index_to_classes.npy',allow_pickle=True).item()
print(index_to_classes)
print('测试集图像数量', len(test_dataset_loader))
print('测试集种类',len(test_dataset_loader.classes))

model=torch.load('../模型训练/model_weight/Resnet50True_batch32_epoch20_33c_20221118.pth')
model=model.eval().to(device)
# 图片路径 以及种类编号
print(test_dataset_loader.imgs[0:10])
img_paths = [img[0] for img in test_dataset_loader.imgs]

df = pd.DataFrame()
df['图片路径']=img_paths
df['图片种类编号']=[img[1] for img in test_dataset_loader.imgs]
df['图片种类名称']=[index_to_classes[index] for index in df['图片种类编号']]
# print(df)

n=5
df_pred=pd.DataFrame()
for idx, row in df.iterrows():
    img_path = row['图片路径']
    img_pil = Image.open(img_path).convert('RGB')
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred, dim=1)  # 对 logit 分数做 softmax 运算
    pred_dict = {}
    top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别

    # top-n 预测结果
    for i in range(1, n + 1):
        pred_dict['top-{}-预测编号'.format(i)] = pred_ids[i - 1]
        pred_dict['top-{}-预测名称'.format(i)] = index_to_classes[pred_ids[i - 1]]
    pred_dict['种类是否存在于预测top_5中(1/0)'] = row['图片种类编号'] in pred_ids
    # 每个类别的预测置信度
    for idx, each in enumerate(list(index_to_classes.values())):
        pred_dict['{}-预测置信度'.format(each)] = pred_softmax[0][idx].cpu().detach().numpy()
    df_pred = df_pred.append(pred_dict, ignore_index=True)

df = pd.concat([df, df_pred], axis=1)
# print(df)
# 预测准确率
acc=sum(df['图片种类名称']==df['top-1-预测名称'])/len(df)
print(acc)
# 预测存在于前n中类别的比例
top_n_acc=sum(df['种类是否存在于预测top_5中(1/0)']==True)/len(df)
print(top_n_acc)
df.to_csv('output/resnet50测试集预测结果.csv', index=False,encoding='utf-8')

fish_df=pd.DataFrame()
acc_list=[]
top_n_acc_list=[]
classes_list=[]
for fish in list(index_to_classes.values()):
    acc_list.append(sum((df['图片种类名称']==fish)&(df['图片种类名称']==df['top-1-预测名称']))/sum(df['图片种类名称']==fish))
    top_n_acc_list.append(sum((df['图片种类名称']==fish)&(df['种类是否存在于预测top_5中(1/0)']==True))/sum(df['图片种类名称']==fish))
    classes_list.append(fish)
fish_df['预测准确率']=acc_list
fish_df['预测存在于前n中类别的比例']=top_n_acc_list
fish_df['种类名称']=classes_list
print(fish_df)
fish_df.to_csv('output/resnet50各种类预测结果.csv',index=False,encoding='utf-8')


