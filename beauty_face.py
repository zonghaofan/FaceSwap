# -*- coding: utf-8 -*-
'''
美肤-磨皮算法
Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
'''
import cv2
import numpy as np
from math import sqrt

def beauty_face(img):
    dst = np.zeros_like(img)
    # int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
    v1 = 3
    v2 = 1
    dx = v1 * 5  # 双边滤波参数之一
    fc = v1 * 12.5  # 双边滤波参数之一
    p = 0.1

    temp4 = np.zeros_like(img)

    temp1 = cv2.bilateralFilter(img, dx, fc, fc)
    temp2 = cv2.subtract(temp1, img)
    temp2 = cv2.add(temp2, (10, 10, 10, 128))
    temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
    temp4 = cv2.add(img, temp3)
    dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
    dst = cv2.add(dst, (10, 10, 10, 255))
    return dst
def seam_less():
    im = cv2.imread('dst_face.jpg')
    obj = cv2.imread('warped_src_face_.jpg')
    # Create an all white mask
    mask=np.load('mask_single_channel.npy')
    print(mask.shape)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    cv2.imshow('mask',mask)
    # The location of the center of the src in the dst
    width, height, channels = obj.shape
    center = (height // 2, width // 2)
    # r = cv2.boundingRect(mask)
    # center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)+2))
    print('center=',center)

    # Seamlessly clone src into dst and put the results in output
    normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
    # mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
    cv2.imshow('normal_clone',normal_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    """beauty"""
    # img = cv2.imread('./imgs/face.png')
    # dst = beauty_face(img)
    # print(dst.shape)
    # cv2.imshow("SRC", img)
    # cv2.imshow("DST", dst)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    """seamless"""
    seam_less()