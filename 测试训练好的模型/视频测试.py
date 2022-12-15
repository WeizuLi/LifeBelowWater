import os
import cv2
import time
import numpy as np
import pandas as pd

from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
font = ImageFont.truetype('simhei.ttf', 32)
index_to_classess = np.load('../模型训练/index_to_classes.npy', allow_pickle=True).item()
print(index_to_classess)
# 导入训练好的模型
model = torch.load('../模型训练/model_weight/resnet18True_batch32_epoch20_33c_20221025.pth')
model = model.eval().to(device)
from torchvision import transforms

# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])


def process_frame(img):
    # 记录该帧开始处理的时间
    start_time = time.time()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 PIL
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    top_n = torch.topk(pred_softmax, 5)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析预测类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析置信度

    # 使用PIL绘制中文
    draw = ImageDraw.Draw(img_pil)
    # 在图像上写字
    for i in range(len(confs)):
        pred_class = index_to_classess[pred_ids[i]]
        text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
        # 文字坐标，中文字符串，字体，bgra颜色
        draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))
    img = np.array(img_pil)  # PIL 转 array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB转BGR

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)
    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
    img = cv2.putText(img, 'FPS  ' + str(int(FPS)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4,
                      cv2.LINE_AA)
    return img


# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0)

# 打开cap
cap.open(0)

# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    if not success:
        print('Error')
        break

    ## !!!处理帧函数
    frame = process_frame(frame)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break

# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()