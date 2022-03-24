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
from PIL import Image 
# from face_landmarks import get_landmark_model, detect_marks
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import streamlit_authenticator as stauth
import subprocess

import hashlib
import mysql.connector
import datetime

with open('static/css/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  #password="Ka$560037KA"
#   password="root"
  password="Uk@336207"
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
        
        
        '''
            Detect Face
        '''
        image = frame.copy()
        success = detect_face(image)
        self.success = success
        if not success :
            cv.putText(frame, "Cannot Find Face", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame
              
        '''
            Get Eye Direction
        '''
        image = frame.copy()
        direction = eye_direction(image)

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

        center_coordinates = (20, 20)
        radius = 5
        thickness = -1
        
        if(state == focused):
            color = (0, 255, 0)
        else: 
            color = (0, 0, 255)

        
                
        cv.circle(frame, center_coordinates, radius, color, thickness)
        cv.putText(frame, direction, (30, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.some_value = state
        
        return frame


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

def clear_form():
    st.session_state["username"] = ""
    st.session_state["password"] = ""
    st.session_state["loggedIn"] = False

def main():
    """Login App"""

    st.sidebar.title("LogIn")
    img=Image.open('assets/images/logo.png')
    st.image(img,width=100)
    st.title("Virtual Immersion Scorer")
    
    # title_container = st.container()
    # col1, col2 = st.columns([1, 20])
    # image = Image.open('assets/images/logo.png')
    # with title_container:
    #     with col1:
    #         st.image(image, width=100)
    #     with col2:
    #         st.title("Virtual Immersion Scorer")


    username = st.sidebar.text_input("User Name", key="username")
    password = st.sidebar.text_input("Password",type='password', key="password")
    if st.sidebar.checkbox("Login", key="loggedIn"):

        if st.button("Logout", on_click=clear_form):
            st.markdown("""
            <style>
            .css-zbg2rx, .css-ex7byz {
                    display: true;
                }
            .css-1cpxqw2 {
                display: none;
            }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
            .css-zbg2rx, .css-ex7byz {
                    display: none;
                }
            .css-1cpxqw2 {
                position: fixed;
                top: 2%;
                right: 2%;
            }
            </style>
            """, unsafe_allow_html=True)
        

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

            # st.title("Real Time Face Emotion Detection Application")
            emotion_count = {'Focused': 0, 'Distracted': 0}
            if "emotion_count" not in st.session_state:
                st.session_state.emotion_count = emotion_count

            st.header("Webcam Live Feed")
            st.write("Click on start to use webcam and detect your face emotion")
            ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer, media_stream_constraints={"video": True, "audio": False})

            logtxtbox = st.empty()
            while ctx.video_processor:
                current_time=datetime.datetime.now()
                duration=(current_time-first_time).seconds
                print()
                # updates only if face found
                if first_time==current_time or duration==2:
                    if ctx.video_processor.success:
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
            st.session_state.emotion_count = {'Focused': 0, 'Distracted': 0}

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