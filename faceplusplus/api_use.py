# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import time
import cv2
import json

#key points
# http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'  # 83 points
# http_url = 'https://api-cn.faceplusplus.com/facepp/v1/face/thousandlandmark'

#face detect
http_url='https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "spqGLj2JzAD7hLhurINlytYKY71nxF7K"
secret = "gBqxuOdBb5N6RiGBCSthGAuAM-HwYSk9"


def get_face_info(http_url, key, secret, filepath):
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    fr = open(filepath, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
    data.append('1')
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append(
        "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
    data.append('--%s--\r\n' % boundary)

    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')

    http_body = b'\r\n'.join(data)
    # build http request
    req = urllib.request.Request(url=http_url, data=http_body)
    # header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

    try:
        # post data to server
        resp = urllib.request.urlopen(req, timeout=5)
        # get response
        qrcont = resp.read()
        # if you want to load as json, you should decode first,
        # for example: json.loads(qrount.decode('utf-8'))
        print(qrcont.decode('utf-8'))
        return json.loads(qrcont.decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(e.read().decode('utf-8'))
        return e.read().decode('utf-8')


def get_face_points(filepath):
    info = get_face_info(http_url, key, secret, filepath)
    print(info)
    image_id = info["image_id"]
    print(image_id)
    face = info['faces'][0]
    points = face['landmark']
    print(len(points))
    face_rectangle=face['face_rectangle']
    mouth_points = []
    eye_points = []
    eyebrow_points = []
    nose_points = []
    contour_points = []
    for i in points:
        if 'mouth' in i:
            mouth_points.append([points[i]['x'], points[i]['y']])
        elif 'eye_' in i:
            eye_points.append([points[i]['x'], points[i]['y']])
        elif 'eyebrow' in i:
            eyebrow_points.append([points[i]['x'], points[i]['y']])
        elif 'nose' in i:
            nose_points.append([points[i]['x'], points[i]['y']])
        else:
            contour_points.append([points[i]['x'], points[i]['y']])
    return mouth_points, eye_points, eyebrow_points, nose_points, contour_points#,\
           #(face_rectangle['left'],face_rectangle['top'],face_rectangle['width']+face_rectangle['left'],face_rectangle['height']+face_rectangle['top'])


if __name__ == '__main__':
    filepath = "../imgs/1.jpg"
    img = cv2.imread(filepath)
    mouth_points, eye_points, eyebrow_points, nose_points, contour_points= get_face_points(filepath)#,face_rectangle\
    print('len(mouth_points)',len(mouth_points))

    for i in mouth_points:
        cv2.circle(img, (tuple(i)), 2, (0, 0, 255), -1)
    for i in eye_points:
        cv2.circle(img, (tuple(i)), 2, (255, 0, 0), -1)
    for i in eyebrow_points:
        cv2.circle(img, (tuple(i)), 2, (0, 255, 0), -1)
    for i in nose_points:
        cv2.circle(img, (tuple(i)), 2, (0, 0, 0), -1)
    for i in contour_points:
        cv2.circle(img, (tuple(i)), 2, (255, 255, 255), -1)
    # cv2.rectangle(img,(face_rectangle[0],face_rectangle[1]),(face_rectangle[2],face_rectangle[3]),(0,0,255))

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




