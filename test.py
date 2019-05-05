from color_transfer.transfer import *
import cv2
def face_color_tranfer():
    path_source='./result1.jpg'
    path_target = './result2.jpg'
    output_path='transfer_img.jpg'

    source_img=cv2.imread(path_source)
    print(source_img.shape)
    target_img = cv2.imread(path_target)
    print(target_img.shape)
    transfer_img = color_transfer(source_img, target_img)
    print(transfer_img.shape)
    cv2.imwrite(output_path,transfer_img)
if __name__ == '__main__':
    face_color_tranfer()
