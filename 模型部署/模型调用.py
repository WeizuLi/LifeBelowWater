import os
import shutil

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torchvision import models, transforms
import torch
from PIL import Image, ImageFont, ImageDraw
import time
import matplotlib

matplotlib.rc("font",family='simhei') # 中文字体
font = ImageFont.truetype('simhei.ttf', 20)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
index_to_classes = np.load('index_to_classes.npy', allow_pickle=True).item()
def predict(image_path='../测试训练好的模型/test_img/3.jpg', video_path=None,option='resnet18'):
    if option == "resnet18":
        model = torch.load('../模型训练/model_weight/Resnet18True_batch32_epoch20_33c_20221118.pth')
    elif option == "resnet50":
        model = torch.load('../模型训练/model_weight/Resnet18True_batch32_epoch20_33c_20221118.pth')
    elif option == "densenet121":
        model = torch.load('../模型训练/model_weight/Densenet121True_batch32_epoch20_33c_20221118.pth')
    # elif option == "LwzModule":
    #     model = torch.load('../模型训练/model_weight/LwzModelTrue_batch32_epoch20_33c_20221118.pth')

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0).to(device)
    model.eval().to(device)
    t1 = time.time()
    out = model(batch_t)
    t2 = time.time()
    # fps = round(float(1 / (t2 - t1)), 3)
    x=index_to_classes.values()
    classes=[w for w in x]
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    width = 0.45  # 柱状图宽度
    y=prob.cpu().detach().numpy()
    plt.figure(figsize=(22, 10))
    ax = plt.bar(x,y , width)

    plt.bar_label(ax, fmt='%.2f', fontsize=15)  # 置信度数值
    plt.tick_params(labelsize=20)  # 设置坐标文字大小

    plt.title(image_path, fontsize=30)
    plt.xticks(rotation=45)  # 横轴文字旋转
    plt.xlabel('类别', fontsize=20)
    plt.ylabel('置信度', fontsize=20)
    # plt.show()
    pred_softmax = torch.nn.functional.softmax(out, dim=1)
    top_n = torch.topk(pred_softmax, 5)  # 取置信度最大的 5个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析出置信度
    draw = ImageDraw.Draw(img)
    for i in range(5):
        class_name =index_to_classes[pred_ids[i]]  # 获取类别名称
        confidence = confs[i] * 100  # 获取置信度
        text = '{:<15} {:>.4f}'.format(class_name, confidence)
        print(text)

        # 文字坐标，中文字符串，字体，rgba颜色
        draw.text((50, 50 + 50 * i), text, font=font, fill=(255, 0, 0, 1))

    if video_path is not None:
        temp_out_dir = time.strftime('%Y%m%d%H%M%S')
        os.mkdir(temp_out_dir)
        print('创建临时文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))
        imgs = mmcv.VideoReader(video_path)
        for frame_id, img in enumerate(imgs):
            ## 处理单帧画面
            img, pred_softmax = pred_single_frame(img,model, n=5)
            # 将处理后的该帧画面图像文件，保存至 /tmp 目录下
            cv2.imwrite(f'{temp_out_dir}/{frame_id:06d}.jpg', img)
        # 把每一帧串成视频文件
        mmcv.frames2video(temp_out_dir, 'output/output2_pred.mp4', fps=imgs.fps, fourcc='MP4V')
        shutil.rmtree(temp_out_dir)  # 删除存放每帧画面的临时文件夹
        print('删除临时文件夹', temp_out_dir)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]],plt,img


def pred_single_frame(img,model, n=5):
    '''
    输入摄像头画面bgr-array，输出前n个图像分类预测结果的图像bgr-array
    '''
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 pil
    input_img = transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = torch.nn.functional.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析出置信度

    # 在图像上写字
    draw = ImageDraw.Draw(img_pil)
    # 在图像上写字
    for i in range(len(confs)):
        pred_class = index_to_classes[pred_ids[i]]
        text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
        # 文字坐标，中文字符串，字体，rgba颜色
        draw.text((50, 50 + 50 * i), text, font=font, fill=(255, 0, 0, 1))
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # RGB转BGR

    return img_bgr, pred_softmax
if __name__ == '__main__':
    result,fps,plt=predict("../测试训练好的模型/test_img/3.jpg",'resnet18')
    print(result)
    print(fps)
    plt.show()
