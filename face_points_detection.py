#! /usr/bin/env python
import dlib
import numpy as np
## Face and points detection
def face_points_detection(img, bbox:dlib.rectangle,forhead_flag=False):
    if forhead_flag:
        PREDICTOR_PATH = 'models/shape_predictor_81_face_landmarks.dat'
    else:
        PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=np.int)

    # return the array of (x, y)-coordinates
    return coords

