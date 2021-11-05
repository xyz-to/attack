import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Attacker import Attacker


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """无目标图片识别攻击"""
    df = pd.read_csv('D://pythonspace//data//labels.csv')
    df = df.loc[:, 'TrueLabels']  # 取行和列
    label_name = pd.read_csv("./data/categories.csv")
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()  # 等会根据这个找类别名字
    img_dir = 'D://pythonspace//data//images'
    attacker = Attacker(img_dir, df)
    adv_examples, final_acc = [], []
    epslion = [0.1, 0.01]
    for i in epslion:
        adv, acc = attacker.fgsm_atteck(i)
        adv_examples.append(adv)
        final_acc.append(acc)

    cnt = 0
    plt.figure(figsize=(30, 30))
    for i in range(len(epslion)):
        for j in range(len(adv_examples[i])):
            cnt += 1
            plt.subplot(len(epslion), len(adv_examples[0] * 2), cnt)    # 显示原始图像
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.ylabel('ep:{}'.format(epslion[i]))
            orig, adv, orig_img, ex = adv_examples[i][j]    # 原始ID，攻击ID，原始图像，攻击图像
            orig_img = np.transpose(orig_img, (1, 2, 0))    # np中更换顺序的方法是transpose,tensor中的是permute
            plt.imshow(orig_img)
            plt.title('orig:{}'.format(label_name[orig].split(',')[0]))
            cnt += 1
            plt.subplot(len(epslion), len(adv_examples[0] * 2), cnt)    # 显示攻击后的图像
            plt.title('adv:{}'.format(label_name[adv].split(',')[0]))
            ex = np.transpose(ex, (1, 2, 0))
            plt.imshow(ex)

    # 显示
    plt.tight_layout()
    plt.show()
