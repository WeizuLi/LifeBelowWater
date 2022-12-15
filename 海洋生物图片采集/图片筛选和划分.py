import os
import random
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm

# 将不是三通道的图片删除
dataset_path = 'dataset'
# for fish in tqdm(os.listdir(dataset_path)):
#     for num in os.listdir(os.path.join(dataset_path, fish)):
#         file_path = os.path.join(dataset_path, fish, num)
#         img = np.array(Image.open(file_path))
#         #图像形状，行数，列数，通道数
#         try:
#             channel = img.shape[2]
#             if channel != 3:
#                 print(file_path, '非三通道，删除')
#                 os.remove(file_path)
#         except:
#             print(file_path, '非三通道，删除')
#             os.remove(file_path)
# 划分数据集和测试集
dataset_name='LifeBelowWater'
classes = os.listdir(dataset_path)
print(classes)
print(len(classes))
# 创建 train 文件夹
os.mkdir(os.path.join(dataset_path, 'train'))
# 创建 test 文件夹
os.mkdir(os.path.join(dataset_path, 'test'))
for fish in classes:
    os.mkdir(os.path.join(dataset_path, 'train', fish))
    os.mkdir(os.path.join(dataset_path, 'test', fish))
# 随机选择图片，按照训练集80%,测试集20%的比例划分
test_pro=0.2
df = pd.DataFrame()
# 遍历每个鱼种类
random.seed(123) # 随机数种子，便于复现
for fish in classes:
    # 将每个鱼类中的图片顺序打乱
    image_list=os.listdir(dataset_path+'/'+fish)
    random.shuffle(image_list)
    # 打乱后按照比例切片
    test_images_list=image_list[:int((len(image_list)*test_pro))]
    train_images_list=image_list[int(len(image_list)*test_pro):]
    # 将图片转移
    i=0
    for img in test_images_list:
        old_path=os.path.join(dataset_path, fish, img)
        # new_path=os.path.join(dataset_path,'test',fish,img)
        new_path=dataset_path+'/test/'+fish+'/'+str(i)+'.jpg'
        i+=1
        shutil.move(old_path,new_path)
    for img in train_images_list:
        old_path=os.path.join(dataset_path, fish, img)
        # new_path=os.path.join(dataset_path,'train',fish,img)
        new_path=dataset_path+'/train/'+fish+'/'+str(i)+'.jpg'
        i+=1
        shutil.move(old_path,new_path)
#     将图片全部转移后，删除文件夹
    if len(os.listdir(dataset_path+'/'+fish))==0:
        shutil.rmtree(dataset_path+'/'+fish)
    df = df.append({'类别': fish, '训练集大小': len(train_images_list), '测试集大小': len(test_images_list)},
                   ignore_index=True)
df['总数']=df['训练集大小']+df['测试集大小']
print(df)
# 生成文件
df.to_csv('output/划分训练集统计.csv',encoding='utf-8',index=False)
shutil.move(dataset_path, dataset_name+'_dataset')
# if len(os.listdir(dataset_path)) == 0:
#     shutil.rmtree(dataset_path)

