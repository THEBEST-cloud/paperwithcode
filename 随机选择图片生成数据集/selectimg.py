##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil

random.seed(10)


def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.2  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.copyfile(fileDir_img + name, tarDir_img + name)
        img_id = name.split(".jpg")[0]
        gt_name = img_id + ".xml"
        shutil.copyfile(fileDir_gt + gt_name, tarDir_gt + gt_name)
    return


if __name__ == '__main__':
    fileDir_img = "../train/image/"  # 源图片文件夹路径
    tarDir_img = '../val/image/'  # 移动到新的文件夹路径
    fileDir_gt = "../train/box/"  # 源图片文件夹路径
    tarDir_gt = '../val/box/'  # 移动到新的文件夹路径
    moveFile(fileDir_img)
















