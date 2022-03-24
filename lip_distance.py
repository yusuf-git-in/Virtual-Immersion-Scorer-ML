import cv2 as cv 
import numpy as np
import mediapipe as mp 
import math

mp_face_mesh = mp.solutions.face_mesh

UPPER_LIP = [13]
LOWER_LIP = [14]

def euclaidean_distance(pt1, pt2) :
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def lip_distance(img) :
    '''
        classify yawn or active

    Parameters
    ----------
    img : np.uint8
        
    Returns
    -------
    state : string
        state ; yawn / talking or silent
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
            
            pt1 = mesh_points[UPPER_LIP][0]
            pt2 = mesh_points[LOWER_LIP][0]
            
            '''
                thresholding 
            '''
            dist = euclaidean_distance(pt1, pt2)
            if dist <= 25.0 and dist >= 6.0 :
                state = 'talking'
                
            elif dist > 25.0 :
                state = 'yawning'
                
            else :
                state = 'silent'
                
        return state
        