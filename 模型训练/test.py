from matplotlib import pyplot as plt
Accuracy=[]
train_Accuracy=[]
with open('output/test.txt','r',encoding='utf-8') as f:
    texts=f.readlines()
    print(texts)
    for text in texts:
        num=float(text[4:20])
        print(num)
        Accuracy.append(num)
with open('output/test2.txt','r',encoding='utf-8') as f:
    texts=f.readlines()
    texts1=texts[0:9]
    texts2=texts[9:30]
    print(texts)
    for text in texts1:
        num=float(text[7:23])
        print(num)
        train_Accuracy.append(num)
    for text in texts2:
        num=float(text[8:23])
        print(num)
        train_Accuracy.append(num)
epoch_list = list(range(1, 30 + 1))
plt.xlim(0, 32)
plt.ylim(0.0, 100)
plt.plot(epoch_list, Accuracy)
plt.plot(epoch_list, train_Accuracy)
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.ylabel("trainAccuracy")
plt.show()