import os

import cv2 as cv2
import numpy as np
import pandas as pd
from PIL import Image

print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
print("PATH:", os.environ.get('PATH'))
# file_save_path = f'C:/Users/adm/PycharmProjects/Life Below Water/海洋生物图片采集/dataset/水母/1.txt'
# f=open(file_save_path,'w')
# f.write('123')
# f.close()

list1=[1,23,2,1,1,3]
list=[]
list.extend(set(list1))
print(list)

# with open('海洋生物名称.txt', 'r', encoding='utf-8') as fname:
#     allList = fname.readlines()
#     class_list = []
#     for list in allList:
#         # 去空格换行符
#         list = list.strip("\n")
#         list = list.strip("")
#         # 去重复
#         list = set(list.split('、'))
#         class_list.extend(list)
#     print(class_list,len(class_list))
set1=set(os.listdir('C:/Users/adm/Desktop/dataset'))
print(set1)
for i in set1:
    os.mkdir(path=('C:/Users/adm/Desktop/dataset_list/'+i))
# with open('海洋生物名称.txt', 'r', encoding='utf-8') as fname:
#     allList = fname.readlines()
#     class_list = []
#     for list in allList:
#         # 去空格换行符
#         list = list.strip("\n")
#         list = list.strip("")
#         # 去重复
#         list = set(list.split(', '))
#         class_list.extend(list)
#     print(class_list)
# print(len(set1))
# dataset_path = 'C:/Users/adm/PycharmProjects/Life Below Water/海洋生物图片采集/dataset/多带蝴蝶鱼/23.jpg'
# file_path=dataset_path
# img = np.array(Image.open(file_path))
# print(img.shape[2])
# df= pd.DataFrame()
# for picture in os.listdir('C:/Users/adm/PycharmProjects/Life Below Water/海洋生物图片采集/LifeBelowWater_dataset/train/墨鱼'):  # 遍历每张图像
#     img = cv2.imread('C:/Users/adm/PycharmProjects/Life Below Water/海洋生物图片采集/LifeBelowWater_dataset/train/墨鱼/'+picture)
#     df.append({'类别':"墨鱼", '图片名':picture, '图像行数':img.shape[0], '图像列数':img.shape[1],'图像通道数':img.shape[2]}, ignore_index=True)
# print(df)
# print(type(df))
# print(df['类别'])
# print(type(df['类别']))
# 保存为csv
