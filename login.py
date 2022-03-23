from multiprocessing import Value
import cv2 as cv
from time import time, sleep
from matplotlib import use
from nbformat import read
import streamlit as st
# from tensorflow import keras
# from keras.models import model_from_json
# from keras.preprocessing.image import img_to_array
from face_detector import detect_face
from eye_tracking import eye_direction
from head_pose_estimation import head_pose 
# from face_landmarks import get_landmark_model, detect_marks
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import streamlit_authenticator as stauth
import subprocess

import hashlib
import mysql.connector
import datetime

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  #password="Ka$560037KA"
  password="root"
)

print(mydb)

cursor = mydb.cursor()
cursor.execute("create database IF NOT EXISTS immersion_scorer_db")
cursor.execute("use immersion_scorer_db")


# # load model
# emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# # load json and create model
# json_file = open('emotion_model1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# classifier = model_from_json(loaded_model_json)

# # load weights into new model
# classifier.load_weights("emotion_model1.h5")

# #load face
# try:
#     face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# except Exception:
#     st.write("Error loading cascade classifiers")
    
    
# emotion_count = {'Focused': 0, 'Distracted': 0}

class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        super().__init__()
        self.some_value = "Focused" 
        self.success = True

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
        
        image = frame
        '''
            Detect Face
        '''
        success = detect_face(image)
        if not success :
            cv.putText(image, "Cannot Find Face", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.success = False
            return image
        
        self.success = True         
        #cv.putText(image, "FOund", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        '''
            Get Eye Direction
        '''
        #eye_direction = eye_direction(image)
        #cv.putText(image, eye_direction, (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        '''
            Head Pose Estimation
        '''
        focused = "Focused"
        distracted = "Distracted"
        state = focused
        pose = head_pose(image)
        if pose == 'forward' :
            state = focused
        else :
            state = distracted

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
                
        cv.circle(image, center_coordinates, radius, color, thickness)
        #cv2.putText(image, state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #if text != "Forward": print(text)
        self.some_value = state
        return image


def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# DB  Functions
def create_usertable():
	cursor.execute('CREATE TABLE IF NOT EXISTS student_table(studentId INT AUTO_INCREMENT PRIMARY KEY,username VARCHAR(20),password VARCHAR(20))')


def add_userdata(username,password):
	cursor.execute('INSERT INTO student_table(username,password) VALUES (?,?)',(username,password))
	cursor.commit()

def login_user(username,password):
	cursor.execute('SELECT * FROM student_table WHERE username = %s AND password = %s',(username,password))
	data = cursor.fetchall()
	return data


def view_all_users():
	cursor.execute('SELECT * FROM student_table')
	data = cursor.fetchall()
	return data

def main():
    """Login App"""

    st.sidebar.title("LogIn")
    st.title("Virtual Immersion Scorer")

    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password",type='password')
    if st.sidebar.checkbox("Login"):
        create_usertable()
        hashed_pswd = make_hashes(password)

        result = login_user(username,check_hashes(password,hashed_pswd))
        if result or True:
            first_time=datetime.datetime.now()
            print(first_time)
            current_time=first_time
            print(current_time)

            # subprocess.Popen(["streamlit", "run", "app.py"])

            print("###############",username,"###############")

            st.title("Real Time Face Emotion Detection Application")
            emotion_count = {'Focused': 0, 'Distracted': 0}
            if "emotion_count" not in st.session_state:
                st.session_state.emotion_count = emotion_count

            st.header("Webcam Live Feed")
            st.write("Click on start to use webcam and detect your face emotion")
            ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

            logtxtbox = st.empty()
            while ctx.video_processor:
                current_time=datetime.datetime.now()
                duration=(current_time-first_time).seconds
                print(duration)
                # updates only if face found
                if first_time==current_time or duration==2 and ctx.video_processor.success:
                    emotion_count[ctx.video_processor.some_value] += 1
                    logtxtbox.write(str(ctx.video_processor.some_value)+"\n"
                        +str(emotion_count['Focused'])+","
                        +str(emotion_count['Distracted']))
                    print(ctx.video_processor.some_value, emotion_count)
                    st.session_state.emotion_count = emotion_count
                    first_time=current_time
                    current_time=datetime.datetime.now()
                
            emotion_count = st.session_state.emotion_count
            print("End:",emotion_count)

            # st.success("Logged In as {}".format(username))

            # task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
            # if task == "Add Post":
            #     st.subheader("Add Your Post")
            # elif task == "Analytics":
            #     st.subheader("Analytics")
            # elif task == "Profiles":
            #     st.subheader("User Profiles")
            #     user_result = view_all_users()
            #     clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
            #     st.dataframe(clean_db)
        else:
            st.warning("Incorrect Username/Password")


if __name__ == '__main__':
    main()