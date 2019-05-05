#! /usr/bin/env python
import os
import cv2
import argparse

import numpy as np

from face_detection import face_detection
from face_points_detection import face_points_detection
from face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points
from color_transfer.transfer import *

eyebrow_flag=True
debug_68_flag=True #show face key points
debug_81_flag=False
transfer_color_flag=True
blur_flag=True
forhead_flag=False
def point_line_side(points):
    """
    three points >0 face in center
    four points >0 face in right
    four points <0 face in left
    :param points:
    :return: face side
    """
    x1, y1 = points[29]
    x2, y2 = points[30]
    symbol=[]
    if x2-x1:
        k = (y2 - y1)/(x2 - x1)
        b = (x2*y1-x1*y2)/(x2-x1)
        #nose points
        for i in points[31:36]:
            side_flag=round(k*i[0]+b-i[1],3)
            print('side_flag=',side_flag)
            symbol.append(side_flag)
    else:
        x=x2
        for i in points[31:36]:
            side_flag=round(i[0]-x,3)
            print('no k,side_flag=',side_flag)
            symbol.append(side_flag)
    if len([i for i in symbol if i>0])<=1:
        symbol_flag = 'left'
    elif len([i for i in symbol if i > 0]) >=4:
        symbol_flag = 'right'
    else:
        symbol_flag =  'center'
    return symbol_flag
def select_face(im, r=10,forhead_flag=False):
    faces = face_detection(im)
    print('len(faces)',len(faces))
    print(type(faces))
    print('faces=',faces)
    if len(faces) == 0:
        print('Detect 0 Face !!!')
        exit(-1)

    if len(faces) == 1:
        bbox = faces[0]
    else:
        bbox = []

        def click_on_face(event, x, y, flags, params):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            for face in faces:
                if face.left() < x < face.right() and face.top() < y < face.bottom():
                    bbox.append(face)
                    break

        im_copy = im.copy()
        # print('im_copy.shape=',im_copy.shape)
        for face in faces:
            # print('face=',face)
            # print(type(face))
            # print('face.left()=',face.left())
            # draw the face bounding box
            cv2.rectangle(im_copy, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)
        cv2.imshow('Click the Face:', im_copy)
        cv2.setMouseCallback('Click the Face:', click_on_face)
        while len(bbox) == 0:
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        bbox = bbox[0]
        print('bbox=',bbox)

    points = np.asarray(face_points_detection(im, bbox,forhead_flag))
    debug_img=im.copy()
    if eyebrow_flag:
        pass
    else:
        #exclude Squat and eyebrow
        points=np.vstack((points[:17],points[27:]))
    # print(points)
    """debug vis detect face points in (0~16) is Squat ,(17~26) is eyebrow"""
    if debug_68_flag:
        for i in points:
            cv2.circle(debug_img,tuple(i),2,(255,0,0),-1)
        cv2.circle(debug_img, tuple(points[27]), 2, (0, 0, 255), -1)
        cv2.circle(debug_img, tuple(points[31]), 2, (0, 255, 0), -1)
        cv2.circle(debug_img, tuple(points[35]), 2, (0, 0, 0), -1)

    im_h, im_w = im.shape[:2]
    left, top = np.min(points, 0)
    #vis left_top and right_bottom point

    right, bottom = np.max(points, 0)


    """debug vis left_top and right_bottom point"""
    if debug_68_flag:
        cv2.circle(debug_img, (left, top), 4, (0, 255, 0), -1)
        cv2.circle(debug_img, (right, bottom), 4, (0, 0, 255), -1)

    cv2.imshow('debug_img',debug_img)
    cv2.waitKey(0)

    x, y = max(0, left-r), max(0, top-r)
    # print('x, y=',x, y)
    print(np.asarray([[x, y]]))
    w, h = min(right+r, im_w)-x, min(bottom+r, im_h)-y

    """debug vis mobile key points"""
    # for i in (points - np.asarray([[x, y]])):
    #     cv2.circle(im, tuple(i), 2, (255, 255, 0), -1)
    # cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 1)
    # cv2.imshow('Click the Face:', im)
    # cv2.waitKey(0)

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]

def select_face_81(im, r=10):
    print('detect 81 points')
    faces = face_detection(im)
    print('len(faces)',len(faces))
    if len(faces) == 0:
        print('Detect 0 Face !!!')
        exit(-1)

    if len(faces) == 1:
        bbox = faces[0]
    else:
        bbox = []

        def click_on_face(event, x, y, flags, params):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            for face in faces:
                if face.left() < x < face.right() and face.top() < y < face.bottom():
                    bbox.append(face)
                    break

        im_copy = im.copy()
        # print('im_copy.shape=',im_copy.shape)
        for face in faces:
            # print('face=',face)
            # print(type(face))
            # print('face.left()=',face.left())
            # draw the face bounding box
            cv2.rectangle(im_copy, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)
        cv2.imshow('Click the Face:', im_copy)
        cv2.setMouseCallback('Click the Face:', click_on_face)
        while len(bbox) == 0:
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        bbox = bbox[0]
        print('bbox=',bbox)

    points = np.asarray(face_points_detection_81(im, bbox))

    if eyebrow_flag:
        pass
    else:
        #exclude Squat and eyebrow
        points=np.vstack((points[:17],points[27:]))
    # print(points)
    """debug vis detect face points in (0~16) is Squat ,(17~26) is eyebrow"""
    if debug_81_flag:
        for i in points:
            cv2.circle(im,tuple(i),2,(255,0,0),-1)

    im_h, im_w = im.shape[:2]
    left, top = np.min(points, 0)
    #vis left_top and right_bottom point

    right, bottom = np.max(points, 0)


    """debug vis left_top and right_bottom point"""
    if debug_81_flag:
        cv2.circle(im, (left, top), 4, (0, 255, 0), -1)
        cv2.circle(im, (right, bottom), 4, (0, 0, 255), -1)


    x, y = max(0, left-r), max(0, top-r)
    # print('x, y=',x, y)
    print(np.asarray([[x, y]]))
    w, h = min(right+r, im_w)-x, min(bottom+r, im_h)-y

    """debug vis mobile key points"""
    # for i in (points - np.asarray([[x, y]])):
    #     cv2.circle(im, tuple(i), 2, (255, 255, 0), -1)
    # cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 1)
    # cv2.imshow('Click the Face:', im)
    # cv2.waitKey(0)

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]
def get_transfer_color(face,points):
    if transfer_color_flag:
        h, w = face.shape[:2]
        mask = mask_from_points((h, w), points, erode_flag=0)
        mask_ = np.mean(face, axis=2) > 0
        mask = np.asarray(mask * mask_, dtype=np.uint8)
        new_face = apply_mask(face, mask)

        return new_face
#高斯双边滤波
def bi_demo(img):
    # dst = cv2.bilateralFilter(src=image, d=9, sigmaColor=75, sigmaSpace=75)
    # dst = cv2.pyrMeanShiftFiltering(src=image, sp=15, sr=20)
    dst = np.zeros_like(img)
    # int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
    v1 = 5
    v2 = 2
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
def get_channels_img(single):
    single = np.expand_dims(single, axis=-1)
    img = np.concatenate((single, single, single), axis=-1)
    return img
def skin_color(warped_face,dst_face):
    # warped_face=cv2.cvtColor(warped_face.astype(np.float32),cv2.COLOR_BGR2HSV)
    # # print('warped_face.shape',warped_face.shape)
    # dst_face=cv2.cvtColor(dst_face,cv2.COLOR_BGR2HSV)
    b_dst = dst_face[..., 0]
    g_dst = dst_face[..., 1]
    r_dst = dst_face[..., 2]

    b_normal = warped_face[..., 0] / np.max(warped_face[..., 0])
    # print('b_normal',b_normal[20:40,20:40])
    g_normal = warped_face[..., 1] / np.max(warped_face[..., 1])
    r_normal = warped_face[..., 2] / np.max(warped_face[..., 2])
    face_ = np.concatenate((np.expand_dims(b_normal * np.max(b_dst), axis=-1),
                            np.expand_dims(g_normal * np.max(g_dst), axis=-1),
                            np.expand_dims(r_normal * np.max(r_dst), axis=-1)), axis=-1)
    # print('face_.shape',face_.shape)
    # face_=cv2.cvtColor(np.asarray(face_).astype(np.float32),cv2.COLOR_HSV2BGR)

    return face_
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--src', required=True, help='Path for source image')
    parser.add_argument('--dst', required=True, help='Path for target image')
    parser.add_argument('--out', required=True, help='Path for storing output images')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    parser.add_argument('--no_debug_window', default=False, action='store_true', help='Don\'t show debug window')
    args = parser.parse_args()
    print(args)

    # Real test
    src_img = cv2.imread(args.src)
    dst_img = cv2.imread(args.dst)

    # Select src face:face key points,points shape,roi face
    src_points, src_shape, src_face = select_face(src_img)
    cv2.imshow('src_face', src_face)


    # print('src_points.shape',src_points.shape)
    # # Select dst face
    dst_points, dst_shape, dst_face = select_face(dst_img)
    # cv2.imwrite('dst_face.jpg',dst_face)
    # print('dst_points.shape',dst_points.shape)
    cv2.imshow('dst_face',dst_face)

    """dst weight change"""
    # debug nose side
    side_flag = point_line_side(dst_points)
    print('side_flag=', side_flag)
    if side_flag=='left':
        new_points = (0.05 * src_points + 0.95 * dst_points)
        dst_points = (new_points * np.max(dst_points) / np.max(new_points)).astype(np.int32)
    # else:
    #     new_points = (0.005 * src_points + 0.995 * dst_points)
    #     dst_points = (new_points * np.max(dst_points) / np.max(new_points)).astype(np.int32)

    # print(dst_points.shape)
    h, w = dst_face.shape[:2]
    """add transfer color"""
    if transfer_color_flag:
        dst_face_new = get_transfer_color(dst_face,dst_points)
        src_face_new = get_transfer_color(src_face, src_points)
        source_img = dst_face_new
        target_img = src_face_new
        src_face = color_transfer(source_img, target_img)
        cv2.imshow("src_face_color", src_face)
        # cv2.imshow('dst_face_new',dst_face_new)
        # cv2.imshow('src_face_new',src_face_new)

    """question"""
    # src_points[16:26,1]=0
    # dst_points[16:26,1]=0
    # src_points[0,1]=0
    # dst_points[0,1]=0

    # Warp Image
    if not args.warp_2d:
        ## 3d warp
        # warped_src_face = warp_image_3d(src_face, src_points[:48], dst_points[:48], (h, w))
        ## 3d warp - luzi with all points
        # cv2.waitKey(0)
        if eyebrow_flag:
        #need to transfer color
            # if side_flag == 'left':
            warped_src_face = warp_image_3d(src_face, src_points[:-20], dst_points[:-20], (h, w))
            # else:
            #     warped_src_face = warp_image_3d(src_face, src_points[:-10], dst_points[:-10], (h, w))

            cv2.imshow("src_face", src_face)
            cv2.imshow("warped_src_face", warped_src_face)
            cv2.imwrite('src_face.jpg', src_face)
            np.save('src_points.npy', src_points)
            np.save('dst_points.npy', dst_points)
        else:
            warped_src_face = warp_image_3d(src_face, src_points[:-20], dst_points[:-20], (h, w))

            cv2.imshow("warped_src_face", warped_src_face)
    else:
        ## 2d warp
        src_mask = mask_from_points(src_face.shape[:2], src_points, erode_flag=0)
        src_face = apply_mask(src_face, src_mask)
        # Correct Color for 2d warp
        if args.correct_color:
            warped_dst_img = warp_image_3d(dst_face, dst_points[:48], src_points[:48], src_face.shape[:2])
            src_face = correct_colours(warped_dst_img, src_face, src_points)
        # Warp
        warped_src_face = warp_image_2d(src_face, transformation_from_points(dst_points, src_points), (h, w, 3))

    ## Mask for blending
    # mask = mask_from_points((h, w), dst_points)
    # [luzi: erode off]
    mask = mask_from_points((h, w), dst_points, erode_flag=0)
    cv2.imshow("mask_dst", mask)
    # cv2.imshow("mask", mask)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask*mask_src, dtype=np.uint8)
    cv2.imshow("mask_fillConvexPoly", mask)
    # cv2.waitKey(0)

    # Correct color
    if not args.warp_2d and args.correct_color:
        warped_src_face = apply_mask(warped_src_face, mask)

        dst_face_masked = apply_mask(dst_face, mask)
        warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
        cv2.imshow('warped_src_face_', warped_src_face)
    if blur_flag:
        warped_src_face=bi_demo(warped_src_face)
        cv2.imshow('warped_src_face_bi', warped_src_face)
    ## Shrink the mask
    kernel = np.ones((7,7), np.uint8)
    mask_single_channel = cv2.erode(mask, kernel, iterations=1)
    # np.save('mask_single_channel.npy',mask_single_channel)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    # cv2.waitKey(0)

    mask_dst=(1- mask_single_channel / np.max(mask_single_channel)).astype(np.uint8)
    mask_dst=get_channels_img(mask_dst)

    mask_warp=get_channels_img(mask_single_channel / np.max(mask_single_channel))

    cv2.imwrite('warped_src_face.jpg', warped_src_face)
    print('warped_src_face.shape',warped_src_face.shape)
    print('dst_face.shape',dst_face.shape)
    warped_src_face=warped_src_face * mask_warp
    warped_src_face_normal=skin_color(warped_src_face,dst_face)
    cv2.imwrite('warped_src_face_normal.jpg', warped_src_face_normal)

    # # Poisson Blending
    # # r = cv2.boundingRect(mask_single_channel)
    # # center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    width, height, channels = warped_src_face.shape
    center = (height // 2, width // 2)


    # output = mask_dst * dst_face + mask_warp*warped_src_face_
    # cv2.imwrite('output.jpg',output)
    # # cv2.imshow('output',output.astype(np.float64))
    output = cv2.seamlessClone(warped_src_face_normal.astype(np.float32), dst_face, mask_single_channel, center, cv2.NORMAL_CLONE)
    # cv2.imshow('output', output.astype(np.float64))
    x, y, w, h = dst_shape
    dst_img_cp = dst_img.copy()
    dst_img_cp[y:y+h, x:x+w] = output
    output = dst_img_cp

    # output=bi_demo(output)
    # cv2.imshow('output', output)

    """merge result apply beautfy all face """

    # out_points, out_shape, out_face = select_face_81(dst_img_cp)
    # cv2.imshow('dst_img_cp', dst_img_cp)
    # h_out,w_out=out_face.shape[:2]
    # mask = mask_from_points((h_out, w_out), out_points, erode_flag=0)
    # mask_ = np.mean(out_face, axis=2) > 0
    # mask = np.asarray(mask * mask_, dtype=np.uint8)
    # kernel = np.ones((10, 10), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # output = apply_mask(out_face, mask)
    # output = bi_demo(output)
    # r = cv2.boundingRect(mask)
    # center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    # output = cv2.seamlessClone(output, out_face, mask, center, cv2.NORMAL_CLONE)
    # cv2.imshow('output', output)
    # x, y, w, h = out_shape
    # dst_img_cp[y:y + h, x:x + w] = output
    # output = dst_img_cp
    #
    # # cv2.imshow('output',output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    dir_path = os.path.dirname(args.out)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    cv2.imwrite(args.out, output)

    # ##For debug
    # if not args.no_debug_window:
    #     cv2.imshow("From", dst_img)
    #     cv2.imshow("To", output)
    #     cv2.waitKey(0)
    #
    #     cv2.destroyAllWindows()
