from multiprocessing import Value
from time import time, sleep
import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from gaze_tracking import GazeTracking
import mediapipe as mp


def eye_on_mask(mask, side, shape):
    """
    Create ROI on mask of the size of eyes and also find the extreme points of each eye

    Parameters
    ----------
    mask : np.uint8
        Blank mask to draw eyes on
    side : list of int
        the facial landmark numbers of eyes
    shape : Array of uint32
        Facial landmarks

    Returns
    -------
    mask : np.uint8
        Mask with region of interest drawn
    [l, t, r, b] : list
        left, top, right, and bottommost points of ROI

    """
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0

    
def contouring(thresh, mid, img, end_points, right=False):
    """
    Find the largest contour on an image divided by a midpoint and subsequently the eye position

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image of one side containing the eyeball
    mid : int
        The mid point between the eyes
    img : Array of uint8
        Original Image
    end_points : list
        List containing the exteme points of eye
    right : boolean, optional
        Whether calculating for right eye or left eye. The default is False.

    Returns
    -------
    pos: int
        the position where eyeball is:
            0 for normal
            1 for left
            2 for right
            3 for up

    """
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass
    
def process_thresh(thresh):
    """
    Preprocessing the thresholded image

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image to preprocess

    Returns
    -------
    thresh : Array of uint8
        Processed thresholded image

    """
    thresh = cv2.erode(thresh, None, iterations=2) 
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh


def print_eye_pos(img, left, right):
    """
    Print the side where eye is looking and display on image

    Parameters
    ----------
    img : Array of uint8
        Image to display on
    left : int
        Position obtained of left eye.
    right : int
        Position obtained of right eye.

    Returns
    -------
    None.

    """
    if left == right and left != 0:
        text = ''
        if left == 1:
            #print('Looking left')
            text = 'Looking left'
        elif left == 2:
            #print('Looking right')
            text = 'Looking right'
        elif left == 3:
            #print('Looking up')
            text = 'Looking up'
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, text, (30, 30), font,  
                   1, (0, 255, 255), 2, cv2.LINE_AA) 

face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

gaze = GazeTracking()


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# emotion_count = {'Focused': 0, 'Distracted': 0}

class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        super().__init__()
        self.some_value = "Focused" 

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        
        #image gray
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(
        #     image=img_gray, scaleFactor=1.3, minNeighbors=5)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(img=img, pt1=(x, y), pt2=(
        #         x + w, y + h), color=(255, 0, 0), thickness=2)
        #     roi_gray = img_gray[y:y + h, x:x + w]
        #     roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        #     if np.sum([roi_gray]) != 0:
        #         roi = roi_gray.astype('float') / 255.0
        #         roi = img_to_array(roi)
        #         roi = np.expand_dims(roi, axis=0)
        #         prediction = classifier.predict(roi)[0]
        #         maxindex = int(np.argmax(prediction))
        #         finalout = emotion_dict[maxindex]
        #         output = str(finalout)
        #     label_position = (x, y)
        #     cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #img = cv2.imread('trial.png')
        # gaze.refresh(frame)

        # frame = gaze.annotated_frame()
        # text = ""

        # if gaze.is_blinking():
        #     text = "Blinking"
        #     #print(text)
        # elif gaze.is_right():
        #     text = "Looking right"
        #     #print(text)
        # elif gaze.is_left():
        #     text = "Looking left"
        #     #print(text)
        # elif gaze.is_center():
        #     text = "Looking center"
        #     #print(text)

        # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)


        image = frame
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False
        
        # Get the result
        results = face_mesh.process(image)
        
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
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

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
            

                # See where the user's head tilting
                focused = "Focused"
                distracted = "Distracted"
                state = focused

                if y < -10:
                    text = "Looking Left"
                    state = distracted
                elif y > 10:
                    text = "Looking Right"
                    state = distracted
                elif x < -10:
                    text = "Looking Down"
                    state = distracted
                elif x > 10:
                    text = "Looking Up"
                    state = distracted
                else:
                    text = "Forward"
                    state = focused

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))

                # Center coordinates
                center_coordinates = (20, 20)
                
                # Radius of circle
                radius = 5
                
                # Blue color in BGR
                if(state == focused):
                    color = (0, 255, 0)
                else: 
                    color = (0, 0, 255)

                thickness = -1
                
                cv2.circle(image, center_coordinates, radius, color, thickness)
                #cv2.putText(image, state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                #if text != "Forward": print(text)
                self.some_value = state
        return image

def main():
    # Face Analysis Application #

    st.title("Real Time Face Emotion Detection Application")

    emotion_count = {'Focused': 0, 'Distracted': 0}
    if "emotion_count" not in st.session_state:
        st.session_state.emotion_count = emotion_count

    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your face emotion")
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    logtxtbox = st.empty()
    while ctx.video_processor:
        emotion_count[ctx.video_processor.some_value] += 1
        logtxtbox.write(str(ctx.video_processor.some_value)+"\n"
            +str(emotion_count['Focused'])+","
            +str(emotion_count['Distracted']))
        # cntbox.write(emotion_count)
        print(ctx.video_processor.some_value, emotion_count)
        st.session_state.emotion_count = emotion_count
    emotion_count = st.session_state.emotion_count
    print("End:",emotion_count)

if __name__ == "__main__":
    main()
