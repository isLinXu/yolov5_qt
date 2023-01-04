import cv2
from numpy import *
import numpy as np
import os
import shutil

path = './Temporary_folder'  # 存放需要图像增强的图片的文件夹
result = os.listdir(path)  # 读取文件夹内文件
#result.sort(key=lambda x:int(x.split('.')[0]))
if not os.path.exists('./train'):
    os.mkdir('./train')
if not os.path.exists('./train/1'):
    os.mkdir('./train/1')
if not os.path.exists('./test'):
    os.mkdir('./test')
if not os.path.exists('./test/1'):
    os.mkdir('./test/1')
num = 0
for i in result:
    num += 1
    if num % 3 == 0:
        shutil.copyfile(path + '/' + i, './train/1' + '/' + i)
    if num % 3 == 1:
        shutil.copyfile(path + '/' + i, './train/1' + '/' + i)
    if num % 3 == 2:
        shutil.copyfile(path + '/' + i, './test/1' + '/' + i)

print('分类完成')

