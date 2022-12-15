import os

import cv2
import pandas as pd
dataset_path = 'C:/Users/adm/PycharmProjects/Life Below Water/海洋生物图片采集/LifeBelowWater_dataset'
df = pd.DataFrame()
for fish in os.listdir(dataset_path+'/train'): # 遍历每个类别
    os.chdir(dataset_path+'/train/'+fish)
    for picture in os.listdir(): # 遍历每张图像
        try:
            img = cv2.imread(picture)
            df = df.append({'类别':fish, '图片名':picture, '图像行数':img.shape[0], '图像列数':img.shape[1],'图像通道数':img.shape[2]}, ignore_index=True)
        except:
            print('读取错误')
    os.chdir(dataset_path+'/train')
os.chdir('../')
for fish in os.listdir(dataset_path+'/test'): # 遍历每个类别
    os.chdir(dataset_path+'/test/'+fish)
    for picture in os.listdir(): # 遍历每张图像
        try:
            img = cv2.imread(picture)
            df = df.append({'类别':fish, '图片名':picture, '图像行数':img.shape[0], '图像列数':img.shape[1],'图像通道数':img.shape[2]}, ignore_index=True)
        except:
            print('读取错误')
    os.chdir(dataset_path+'/train')
os.chdir('../')
print(df)
# 保存到本地
df.to_csv('../output/图片大小统计.csv',index=False,encoding='utf-8')