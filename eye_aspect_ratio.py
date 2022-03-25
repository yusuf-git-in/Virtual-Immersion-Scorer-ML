import cv2 as cv 
import numpy as np
import mediapipe as mp 
import math

mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
RIGHT_IRIS = [474,475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

def euclaidean_distance(pt1, pt2) :
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def eye_aspect_ratio(img) :
    '''
        classify drowsy or active

    Parameters
    ----------
    img : np.uint8
        
    Returns
    -------
    state : string
        state ; drowsy or active
    '''
    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
        ) as face_mesh:
        
        img = cv.flip(img, 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        results = face_mesh.process(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        state = ''
        
        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            '''
                Low - 1.0
                High - 16.0
                Threshold - 25% of High ie 6.5
            '''
            left_eye_euclaidean_distance = euclaidean_distance(mesh_points[LEFT_EYE[12]], mesh_points[LEFT_EYE[4]])
            right_eye_euclaidean_distance = euclaidean_distance(mesh_points[RIGHT_EYE[12]], mesh_points[RIGHT_EYE[4]])
            
            avg_euclaidean_distance = (left_eye_euclaidean_distance + right_eye_euclaidean_distance) / 2
             
            if avg_euclaidean_distance <= 6.5 :
                state  = 'drowsy' 
            
            else :
                state = 'active'
            
        return state
        