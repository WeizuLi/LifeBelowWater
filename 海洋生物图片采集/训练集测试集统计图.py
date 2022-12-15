
import matplotlib.pyplot as plt
import matplotlib
# 类别,训练集大小,测试集大小,总数
with open('output/划分训练集统计.csv', 'r', encoding='utf-8') as f:
    # 去除第一行
    all_list=f.readlines()[1:]
    print(all_list)
    train_list =[]
    test_list =[]
    total_list=[]
    classes_list=[]
    for ele in all_list:
        ele=ele.strip('\n')
        ele=ele.split(',')
        classes_list.append(ele[0])
        train_list.append(int(ele[1]))
        test_list.append(int(ele[2]))
        total_list.append(int(ele[3]))
    print(classes_list)
    print(train_list)
    print(test_list)
    print(total_list)

matplotlib.rc("font",family='SimHei') # 中文字体
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(22, 7))
x = classes_list
y1 = test_list
y2 = train_list
plt.xticks(rotation=90) # 横轴文字旋转

plt.bar(x, y1,width=0.50, label='测试集')
plt.bar(x, y2,width=0.50, label='训练集', bottom=y1)

plt.xlabel('类别', fontsize=20)
plt.ylabel('图片数量', fontsize=20)
plt.tick_params(labelsize=13) # 设置坐标文字大小

plt.legend(fontsize=16) # 图例

# 保存为高清的 pdf 文件
plt.savefig('output/各类别图片数量.pdf', dpi=120, bbox_inches='tight')

plt.show()