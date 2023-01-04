import cv2
import numpy as np
from matplotlib import pyplot as plt

def resizeimg(img):
    height, width, channels = img.shape
    if width > 1500 or width < 600:
        scale = 1200 / width
        print("图片的尺寸由 %dx%d, 调整到 %dx%d" % (width, height, width * scale, height * scale))
        scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        return scaled,scale

image=cv2.imread('./1.jpg')
list = []
while 1 :
    #image,scale1 = resizeimg(image)
    HSV = image.copy()
    HSV2 = image.copy()
    num = 0
    def getpos(event,x,y,flags,param):
        global HSV,HSV2
        HSV3 = HSV2.copy()
        if event==cv2.EVENT_MOUSEMOVE:
            HSV = HSV3
            cv2.line(HSV,(0,y),(HSV.shape[1]-1,y),(0,0,0),1,4)
            cv2.line(HSV, (x, 0), (x, HSV.shape[0] - 1), (0, 0, 0), 1, 4)
            cv2.imshow("imageHSV", HSV)
        elif event==cv2.EVENT_LBUTTONDOWN: #定义一个鼠标左键按下去的事件
            HSV = HSV3
            list.append([int(x),int(y)])
            print(list[-1])

    cv2.imshow("imageHSV",HSV)
    #cv2.imshow('image',image)
    cv2.setMouseCallback("imageHSV",getpos)
    cv2.waitKey(0)
