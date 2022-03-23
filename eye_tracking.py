from operator import rshift
import cv2 as cv 
import numpy as np
import mediapipe as mp 

mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
RIGHT_IRIS = [474,475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

def eye_direction(img) :
    '''
        Track Eye Movement

    Parameters
    ----------
    img : np.uint8
        Image to draw faces on
        
    Returns
    -------
    eye_direction : string
        Direction of gaze
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
        direction = 'forward'
        
        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            eye_iris_dict = {
                "li_cx" : 0,
                "li_cy" : 0,
                "ri_cx" : 0,
                "ri_cy" : 0,
                "le_cx" : 0,
                "re_cx" : 0,
                "re_cy" : 0,
                "li_radius" : 0,
                "ri_radius" : 0,
                "le_radius" : 0,
                "le_radius" : 0,
                "center_left_iris" : 0,
                "center_right_iris" : 0,
                "center_left_eye" : 0,
                "center_right_eye" : 0
            }
            
            (eye_iris_dict["li_cx"], eye_iris_dict["li_cy"]), eye_iris_dict["li_radius"] = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (eye_iris_dict["ri_cx"], eye_iris_dict["ri_cy"]), eye_iris_dict["ri_radius"] = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            (eye_iris_dict["le_cx"], eye_iris_dict["le_cy"]), eye_iris_dict["le_radius"] = cv.minEnclosingCircle(mesh_points[LEFT_EYE])
            (eye_iris_dict["re_cx"], eye_iris_dict["re_cy"]), eye_iris_dict["re_radius"] = cv.minEnclosingCircle(mesh_points[RIGHT_EYE])

            eye_iris_dict["center_left_iris"] = np.array([eye_iris_dict["li_cx"], eye_iris_dict["li_cy"]], dtype=np.int32)
            eye_iris_dict["center_right_iris"] = np.array([eye_iris_dict["ri_cx"], eye_iris_dict["ri_cy"]], dtype=np.int32)
            eye_iris_dict["center_left_eye"] = np.array([eye_iris_dict["le_cx"], eye_iris_dict["le_cy"]], dtype=np.int32)
            eye_iris_dict["center_right_eye"] = np.array([eye_iris_dict["re_cx"], eye_iris_dict["re_cy"]], dtype=np.int32)
            
            #cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            #cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)
            #cv.circle(frame, center_left_eye, int(le_radius), (0,255,0), 1, cv.LINE_AA)
            #cv.circle(frame, center_right_eye, int(re_radius), (0,255,0), 1, cv.LINE_AA)
            #cv.putText(frame, str(abs(le_cx-li_cx)), (20, 20), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            #cv.putText(frame, str(abs(re_cx-ri_cx)), (80, 20), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            
            if (abs(eye_iris_dict["le_cx"]-eye_iris_dict["li_cx"]) >= 4.5) :
                 direction = 'left'
        
            if (abs(eye_iris_dict["re_cx"]-eye_iris_dict["ri_cx"]) >= 4.5):
                direction = 'right'
            
        return direction
        