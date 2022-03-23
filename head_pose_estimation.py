import cv2 as cv
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def head_pose(img):
    '''
        Head Pose Estimation

    Parameters
    ----------
    img : np.uint8
        Image to draw faces on
        
    Returns
    -------
    pose : string
        Head pose estimation
    '''
    img = cv.cvtColor(cv.flip(img, 1), cv.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = face_mesh.process(img)    
    img.flags.writeable = True    
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    pose = ''
    
    img_h, img_w, img_c = img.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])       
                

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv.Rodrigues(rot_vec)


            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            if y < -20:
                pose = "left"
    
            elif y > 20:
                pose = "right"
    
            elif x < -20:
                pose = "down"
    
            elif x > 20:
                pose = "up"
    
            else:
                pose = "forward" 
                
    return pose