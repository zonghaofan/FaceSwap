#cdoing:utf-8
import cv2
import numpy as np
import argparse
from faceplusplus.api_use import *
from face_detection import face_detection
from face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points
import os

debug_68_flag=True
def select_face(im,filepath,r=10,forhead_flag=False):
    faces = face_detection(im)
    print('len(faces)',len(faces))
    if len(faces) == 0:
        print('Detect 0 Face !!!')
        exit(-1)

    if len(faces) == 1:
        face_num=1
        bbox = faces[0]
    else:
        filepath = './1.jpg'
        face_num=2
        bbox = []

        def click_on_face(event, x, y, flags, params):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            for face in faces:
                global pan_h,pan_w
                if face.left() < x < face.right() and face.top() < y < face.bottom():
                    bbox.append(face)
                    img_face = im[(face.top() - 15):(face.bottom() + 15), (face.left() - 10):(face.right() + 10)]
                    cv2.imwrite(filepath, img_face)
                    pan_h = face.top() - 15
                    pan_w = face.left() - 10
                    break

        im_copy = im.copy()
        for face in faces:
            # draw the face bounding box
            cv2.rectangle(im_copy, (face.left()-10, face.top()-15), (face.right()+10, face.bottom()+15), (0, 0, 255), 1)

        cv2.imshow('Click the Face:', im_copy)
        cv2.setMouseCallback('Click the Face:', click_on_face)
        while len(bbox) == 0:
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        bbox_ = bbox[0]
        print('bbox_=',bbox_)


    mouth_points, eye_points, eyebrow_points, nose_points, contour_points = get_face_points(filepath)
    points=np.vstack((eye_points, eyebrow_points, nose_points, contour_points,mouth_points))
    print(bbox)
    if face_num>1:
        points+=(pan_w,pan_h)
        print('pan_w,pan_h',pan_w,pan_h)
        print('points.shape',points.shape)

    # points = np.asarray(face_points_detection(im, bbox,forhead_flag))
    debug_img=im.copy()
    # if eyebrow_flag:
    #     pass
    # else:
    #     #exclude Squat and eyebrow
    #     points=np.vstack((points[:17],points[27:]))
    # """debug vis detect face points in (0~16) is Squat ,(17~26) is eyebrow"""
    if debug_68_flag:
        for i in points:
            cv2.circle(debug_img,tuple(i),2,(255,0,0),-1)

    im_h, im_w = im.shape[:2]
    left, top = np.min(points, 0)
    #vis left_top and right_bottom point

    right, bottom = np.max(points, 0)


    """debug vis left_top and right_bottom point"""
    if debug_68_flag:
        cv2.circle(debug_img, (left, top), 4, (0, 255, 0), -1)
        cv2.circle(debug_img, (right, bottom), 4, (0, 0, 255), -1)
    #
    cv2.imshow('debug_img',debug_img)

    x, y = max(0, left-r), max(0, top-r)
    # print('x, y=',x, y)
    print(np.asarray([[x, y]]))
    w, h = min(right+r, im_w)-x, min(bottom+r, im_h)-y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]

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

    src_points, src_shape, src_face =select_face(src_img,args.src)
    cv2.imshow('src_face', src_face)
    cv2.waitKey(0)
    print('src_points.shape', src_points.shape)
    # Select dst face
    dst_points, dst_shape, dst_face = select_face(dst_img,args.dst)
    cv2.imshow('dst_face', dst_face)
    """weight points"""
    new_points = (0.05 * src_points + 0.95 * dst_points)
    dst_points = (new_points * np.max(dst_points) / np.max(new_points)).astype(np.int32)

    h, w = dst_face.shape[:2]
    # Warp Image
    if not args.warp_2d:
        ## 3d warp
        # warped_src_face = warp_image_3d(src_face, src_points[:48], dst_points[:48], (h, w))
        ## 3d warp - luzi with all points
        # need to transfer color
        # if side_flag == 'left':
        warped_src_face = warp_image_3d(src_face, src_points[:-18], dst_points[:-18], (h, w))
        # else:
        #     warped_src_face = warp_image_3d(src_face, src_points[:-10], dst_points[:-10], (h, w))
        cv2.imshow("warped_src_face", warped_src_face)
    else:
        ## 2d warp
        src_mask = mask_from_points(src_face.shape[:2], src_points, erode_flag=0)
        src_face = apply_mask(src_face, src_mask)
        # Correct Color for 2d warp
        if args.correct_color:
            warped_dst_img = warp_image_3d(dst_face, dst_points[:18], src_points[:18], src_face.shape[:2])
            src_face = correct_colours(warped_dst_img, src_face, src_points)
        # Warp
        warped_src_face = warp_image_2d(src_face, transformation_from_points(dst_points, src_points), (h, w, 3))

    mask = mask_from_points((h, w), dst_points, erode_flag=0)
    cv2.imshow("mask_dst", mask)
    # cv2.imshow("mask", mask)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask * mask_src, dtype=np.uint8)
    cv2.imshow("mask_fillConvexPoly", mask)
    # cv2.waitKey(0)

    # Correct color
    if not args.warp_2d and args.correct_color:
        warped_src_face = apply_mask(warped_src_face, mask)

        dst_face_masked = apply_mask(dst_face, mask)
        warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
        cv2.imshow('warped_src_face_', warped_src_face)
    # if blur_flag:
    #     warped_src_face = bi_demo(warped_src_face)
    #     cv2.imshow('warped_src_face_bi', warped_src_face)
    ## Shrink the mask
    kernel = np.ones((10,10), np.uint8)
    mask_single_channel = cv2.erode(mask, kernel, iterations=1)
    # np.save('mask_single_channel.npy',mask_single_channel)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    # cv2.waitKey(0)

    mask_dst = (1 - mask_single_channel / np.max(mask_single_channel)).astype(np.uint8)
    mask_dst = get_channels_img(mask_dst)

    mask_warp = get_channels_img(mask_single_channel / np.max(mask_single_channel))

    warped_src_face = warped_src_face * mask_warp
    warped_src_face_normal = skin_color(warped_src_face, dst_face)

    # # Poisson Blending
    # # r = cv2.boundingRect(mask_single_channel)
    # # center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    width, height, channels = warped_src_face.shape
    center = (height // 2, width // 2)

    # output = mask_dst * dst_face + mask_warp*warped_src_face_
    # cv2.imwrite('output.jpg',output)
    # # cv2.imshow('output',output.astype(np.float64))
    output = cv2.seamlessClone(warped_src_face_normal.astype(np.float32), dst_face, mask_single_channel, center,
                               cv2.NORMAL_CLONE)
    # cv2.imshow('output', output.astype(np.float64))
    x, y, w, h = dst_shape
    dst_img_cp = dst_img.copy()
    dst_img_cp[y:y + h, x:x + w] = output
    output = dst_img_cp

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    dir_path = os.path.dirname(args.out)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    cv2.imwrite(args.out, output)