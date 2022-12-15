import os
import tempfile

import mmcv
import numpy as np
import streamlit as st
from PIL import Image
import magic

from 模型调用 import predict
import time
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("LifeBelowWater")
st.title("水下生物识别")
st.write("")
st.write("")
option = st.selectbox(
     '请选择模型：',
     ('resnet18','resnet50','densenet121', ))
""
option2 = st.selectbox(
     '尝试下列图片：',
     ('image_水母', 'image_小丑鱼','image_寄居蟹'))


file_up = st.file_uploader("上传你的图片（jpeg格式）", type="jpeg")
# file_up = st.file_uploader("上传你的图片（jpeg格式）", type="jpg")
class_to_idx = np.load('class_to_idx.npy', allow_pickle=True).item()
if file_up is None:

    if option2 =="image_水母":
        image=Image.open("../测试训练好的模型/test_img/3.jpg")
        file_up="../测试训练好的模型/test_img/3.jpg"
    elif option2=='image_小丑鱼':
        image=Image.open("../测试训练好的模型/test_img/20.jpg")
        file_up="../测试训练好的模型/test_img/20.jpg"
    else:
        image=Image.open("../测试训练好的模型/test_img/0.jpg")
        file_up="../测试训练好的模型/test_img/0.jpg"
    st.image(image, caption='图片已经加载.', use_column_width=True)
    st.write("")
    st.write("")
    labels,plt,img = predict(file_up,None, option)
    # print out the top 5 prediction labels with scores
    st.success('识别完成~')
    for i in labels:
        st.write(f"Prediction (index:{class_to_idx[i[0]]},class: {i[0]})", ",  置信度: ", i[1])
    st.write("")
    f=plt.gcf()
    file_up='../测试训练好的模型/output/pred'+file_up[-5]+'.jpg'
    f.savefig(file_up)
    image = Image.open(file_up)
    st.image(image,caption='预测置信度.', use_column_width=True)
    st.write("")
    st.image(img,caption='预测置信度.', use_column_width=True)


else:
    st.write(file_up)
    image = Image.open(file_up)
    st.image(image, caption='图片加载完成.', use_column_width=True)
    st.write("")
    st.write("请稍后...")
    labels,plt,img = predict(file_up,None,option)

    # print out the top 5 prediction labels with scores
    st.success('识别完成~')
    for i in labels:
        st.write(f"Prediction (index:{class_to_idx[i[0]]},class: {i[0]})", ",  置信度: ", i[1])
    st.write("")

    f = plt.gcf()
    f.savefig("../测试训练好的模型/output/1.jpg")
    image = Image.open("../测试训练好的模型/output/1.jpg")
    st.image(image, caption='预测置信度.', use_column_width=True)
    st.write("")
    st.image(img, caption='预测置信度.', use_column_width=True)


video_up = st.file_uploader("上传你的视频(mp4格式)",type='mp4')

if video_up is None:
    st.write("您可以选择上传视频")
    option3 = st.selectbox(
        '选择查看预测视频效果：',
        ('output_pred.mp4',))
    if option3=='output_pred.mp4':
        # video=open('output/'+option3,'rb').read()
        # st.video(video)
        video_file=mmcv.VideoReader('output/'+option3)
        pictures = st.slider('预测视频的每一帧', 1, len(video_file))
        picture = pictures - 1
        st.image(video_file[picture],caption='图片加载完成', use_column_width=True)
else:
    # 预测视频，并保存
    predict(file_up,'input/'+video_up.name,option)
    st.write('预测视频已经保存到output\output_pred2.mp4')
if os.path.exists('output/output2_pred.mp4') and video_up is not None:
    st.write(video_up)
    video_bytes=video_up.read()
    st.video(video_bytes)
    video_file = mmcv.VideoReader('output/output2_pred.mp4')
    pictures = st.slider('预测视频的每一帧', 1, len(video_file), key=2)
    picture = pictures - 1
    st.image(video_file[picture], caption='图片加载完成', use_column_width=True)





