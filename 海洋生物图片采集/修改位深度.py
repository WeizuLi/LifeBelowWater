import os

from PIL import Image

path = r'C:\Users\adm\PycharmProjects\Life Below Water\测试训练好的模型\test_img\1.png'
newpath = r'C:\Users\adm\PycharmProjects\Life Below Water\测试训练好的模型\test_img'


def picture(path):
    files = os.listdir(path)
    # for label in files:
    #     # os.mkdir(os.path.join(newpath, label))
    #     for i in os.listdir(os.path.join(path,label)):
    #         files = os.path.join(path, label,i)
    #         img = Image.open(files).convert('RGB')
    #         dirpath = newpath
    #         file_name, file_extend = os.path.splitext(i)
    #         dst = os.path.join(os.path.abspath(dirpath), label,file_name + '.jpg')
    #         img.save(dst)
# picture(path)
img=Image.open(path).convert('RGB')
dirpath=newpath
dst=os.path.join(os.path.abspath(dirpath) ,'1.jpeg')
img.save(dst)
