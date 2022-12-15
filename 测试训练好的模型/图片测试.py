import torch
import torchvision
import torch.nn.functional as F

import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='simhei') # 中文字体
# 导入中文字体，指定字号
font = ImageFont.truetype('simhei.ttf', 14)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

index_to_classess = np.load('../模型训练/index_to_classes.npy', allow_pickle=True).item()
print(index_to_classess)

# 导入训练好的模型
model = torch.load('../模型训练/model_weight/resnet18True_batch32_epoch20_33c_20221025.pth')
model = model.eval().to(device)

from torchvision import transforms
# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])
img_path = 'test_img/9.jpg'
img_pil = Image.open(img_path)
print(np.array(img_pil).shape)
input_img = test_transform(img_pil) # 预处理
print(input_img.shape)
input_img = input_img.unsqueeze(0).to(device)
print(input_img.shape)
pred_logits = model(input_img)
pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算
print(pred_softmax)
plt.figure(figsize=(22, 10))

x = index_to_classess.values()
# x= {0:'水母',1:'虾',2:'螃蟹',3:'鲟鱼',4:'鳄鱼',5:'鳗鱼',6:'黄尾副刺尾鱼'}.values()
y = pred_softmax.cpu().detach().numpy()[0] * 100
width = 0.45 # 柱状图宽度

ax = plt.bar(x, y, width)

plt.bar_label(ax, fmt='%.2f', fontsize=15) # 置信度数值
plt.tick_params(labelsize=20) # 设置坐标文字大小

plt.title(img_path, fontsize=30)
plt.xticks(rotation=45) # 横轴文字旋转
plt.xlabel('类别', fontsize=20)
plt.ylabel('置信度', fontsize=20)
plt.show()

n = 5
top_n = torch.topk(pred_softmax, n) # 取置信度最大的 n 个结果
pred_ids = top_n[1].cpu().detach().numpy().squeeze() # 解析出类别
confs = top_n[0].cpu().detach().numpy().squeeze() # 解析出置信度

print(pred_ids)
print(confs)
draw = ImageDraw.Draw(img_pil)
# dict={0:'水母',1:'虾',2:'螃蟹',3:'鲟鱼',4:'鳄鱼',5:'鳗鱼',6:'黄尾副刺尾鱼'}
for i in range(n):
    class_name = index_to_classess[pred_ids[i]]  # 获取类别名称
    confidence = confs[i] * 100  # 获取置信度
    text = '{:<15} {:>.4f}'.format(class_name, confidence)
    print(text)

    # 文字坐标，中文字符串，字体，rgba颜色
    draw.text((50, 50 + 50 * i), text, font=font, fill=(255, 0, 0, 1))
plt.imshow(img_pil)

plt.show()
img_pil.save('output/9.jpg')
