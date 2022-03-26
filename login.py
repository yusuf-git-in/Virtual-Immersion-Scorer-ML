from multiprocessing import Value
import cv2 as cv
from time import time, sleep
from matplotlib import use
from mysqlx import Result
from nbformat import read
import streamlit as st
# from tensorflow import keras
# from keras.models import model_from_json
# from keras.preprocessing.image import img_to_array
from face_detector import detect_face
from eye_tracking import eye_direction
from head_pose_estimation import head_pose
from eye_aspect_ratio import eye_aspect_ratio
from lip_distance import lip_distance
from PIL import Image 
#from face_landmarks import get_landmark_model, detect_marks
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import streamlit_authenticator as stauth
import subprocess
import streamlit.components.v1 as components
from engagement import lip
import hashlib
import mysql.connector
import datetime


img = Image.open('assets/images/logo.png')
st.set_page_config(page_title="Mimiric: A tool for ML data",
                   page_icon=img,
                #    layout='centered',
                   initial_sidebar_state='auto')
st.markdown(""" 
<nav id="navbar">""",unsafe_allow_html=True)

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


class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        super().__init__()
        self.some_value = "Focused" 
        self.success = True
        self.direction = 0
        self.eye_state = 0
        self.lip_state = 0
        self.pose = 0
        self.engagement_level = 0



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
        self.direction = eye_direction(image)

        '''
            Eye Aspect Ratio
        '''
        image = frame.copy()
        self.eye_state = eye_aspect_ratio(image)
        
        '''
            Lip Distance
        '''
        image = frame.copy()
        self.lip_state = lip_distance(image)
        
        '''
            Head Pose Estimation
        '''
        image = frame.copy()
        focused = "Focused"
        distracted = "Distracted"
        state = focused
        self.pose = head_pose(image)
        
        if self.pose == 'forward' :
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
        cv.putText(frame, "head state "+ self.pose, (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(frame, "eye state "+ self.eye_state, (30,80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv.putText(frame, "lip state "+ self.lip_state, (30,100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv.putText(frame, "eye direction "+ self.direction, (30,140), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        self.some_value = state
        self.engagement_level = lip.calc_delta_new(self.pose, self.eye_state, self.direction, self.lip_state, "Neutral")
        cv.putText(frame, "engagement: "+ str(self.engagement_level), (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame


def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return password
	return False

# DB  Functions
def create_usertable():
	cursor.execute('CREATE TABLE IF NOT EXISTS student(s_id INT NOT NULL PRIMARY KEY,name VARCHAR(255),password VARCHAR(255))')


def add_userdata(sid,username,password):
	cursor.execute('INSERT INTO student(sid,name,password) VALUES (?,?,?)',(sid,username,password))
	cursor.commit()

def login_user(username,password):
	cursor.execute('SELECT * FROM student WHERE name = %s AND password = %s',(username,password))
	data = cursor.fetchall()
	return data


def view_all_users():
	cursor.execute('SELECT * FROM student')
	data = cursor.fetchall()
	return data

def clear_form():
    st.session_state["username"] = ""
    st.session_state["password"] = ""
    st.session_state["loggedIn"] = False
    # st.session_state["sidebarLoginCbx"] = False
    st.session_state["participate"] = False

def main():
    """Login App"""

    st.sidebar.title("LogIn")
    img=Image.open('assets/images/logo.png')
    st.image(img,width=100)
    st.title("Virtual Immersion Scorer")

    username = st.sidebar.text_input("User Name", key="username")
    password = st.sidebar.text_input("Password",type='password', key="password")
    sidebarLoginCbx = st.sidebar.checkbox("Login", key="loggedIn")
    # if not sidebarLoginCbx in st.session_state:
    #     st.session_state.sidebarLoginCbx = sidebarLoginCbx
    if sidebarLoginCbx:

        # if st.button("Logout", on_click=clear_form):
        #     st.markdown("""
        #     <style>
        #     .css-zbg2rx, .css-ex7byz {
        #             display: true;
        #         }
        #     .css-1cpxqw2 {
        #         display: none;
        #     }
        #     </style>
        #     """, unsafe_allow_html=True)
        # else:
        #     st.markdown("""
        #     <style>
        #     .css-zbg2rx, .css-ex7byz {
        #             display: none;
        #         }
        #     .css-1cpxqw2 {
        #         position: fixed;
        #         top: 2%;
        #         right: 2%;
        #         z-index: 2;
        #     }
        #     </style>
        #     """, unsafe_allow_html=True)
        

        create_usertable()
        hashed_pswd = make_hashes(password)

        result = login_user(username,check_hashes(password,hashed_pswd))
        for res in result:
            print("$$$$$$$$$$$$$$$$$$$$$$$",res,"$$$$$$$$$$$$$$$$$$$$$$$$$")
        if len(result)>0:
            st.title("Hi, "+str(username))
        if len(result)>0:

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
                    z-index: 2;
                }
                </style>
                """, unsafe_allow_html=True)

            courseList = ["","Course1","Course2"]
            courseCBx = st.selectbox(label="Choose course",options=courseList)
            if courseCBx:
                first_time=datetime.datetime.now()
                current_time=first_time

                print("###############",username,"###############")

                # st.title("Real Time Face Emotion Detection Application")
                #emotion_count = {'Focused': 0, 'Distracted': 0}
                engagement_level = 0
                chkClicked = False
                handRaisedCount = 0
                # if "emotion_count" not in st.session_state:
                #     st.session_state.emotion_count = emotion_count

                if engagement_level not in st.session_state:
                    st.session_state.engagement_level = engagement_level
                
                if "chkStClicked" not in st.session_state:
                    st.session_state.chkStClicked = chkClicked

                if "raisedCnt" not in st.session_state:
                    st.session_state.raisedCnt = handRaisedCount

                st.header("Webcam Live Feed")
                st.write("Click on start to use webcam and detect your face emotion")
                ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer, media_stream_constraints={"video": True, "audio": False})

                participateCbx = st.checkbox(label="Participate", key="participate")
                components.html("""
                    <script>
                    const checkBxText = window.parent.document.querySelectorAll('.css-1djdyxw')
                    checkBxText[1].style.position = 'relative'
                    checkBxText[1].style.paddingRight = '120px'
                    </script>
                    """)
                logtxtbox = st.empty()
                checked = st.empty()
                while ctx.video_processor:
                    current_time=datetime.datetime.now()
                    duration=(current_time-first_time).seconds
                    print()
                    # updates only if face found
                    if first_time == current_time or duration == 2:
                        # if ctx.video_processor.success:
                        #     emotion_count[ctx.video_processor.some_value] += 1
                            
                        # logtxtbox.write(str(ctx.video_processor.some_value)+"\n"
                        #     +str(emotion_count['Focused'])+","
                        #     +str(emotion_count['Distracted']))
                        logtxtbox.write(str(ctx.video_processor.engagement_level) + "\n" +str(engagement_level))
                        # print(ctx.video_processor.some_value, emotion_count)
                        # st.session_state.emotion_count = emotion_count
                        print(ctx.video_processor.engagement_level, engagement_level)
                        st.session_state.engagement_level = engagement_level
                        first_time=current_time
                        current_time=datetime.datetime.now()

                    if participateCbx and (not chkClicked):
                        chkClicked = True
                        handRaisedCount = st.session_state.raisedCnt
                        handRaisedCount += 1

                        st.session_state.raisedCnt = handRaisedCount
                        st.session_state.chkStClicked = chkClicked

                        checked.write('Hand Raised:: Count: '+str(handRaisedCount))

                        components.html("""
                            <script>
                            const checkBxText = window.parent.document.querySelectorAll('.css-1djdyxw')
                            checkBxText[1].style.border = '1px solid #f94144'
                            </script>
                            """)
                    elif (not participateCbx) and chkClicked:
                        chkClicked = False

                        st.session_state.raisedCnt = handRaisedCount
                        st.session_state.chkStClicked = chkClicked

                        checked.write('Hand Lowered:: Count: '+str(handRaisedCount))

                        components.html("""
                            <script>
                            const checkBxText = window.parent.document.querySelectorAll('.css-1djdyxw')
                            checkBxText[1].style.border = '1px solid black'
                            </script>
                            """)
                    
                    chkClicked = st.session_state.chkStClicked
                    handRaisedCount = st.session_state.raisedCnt
                    st.session_state.raisedCnt = handRaisedCount

                    engagement_level = st.session_state.engagement_level
                    st.session_state.engagement_level = engagement_level
                    
                    if participateCbx and (not chkClicked):
                        chkClicked = True
                        handRaisedCount = st.session_state.raisedCnt
                        handRaisedCount += 1

                        st.session_state.raisedCnt = handRaisedCount
                        st.session_state.chkStClicked = chkClicked

                        checked.write('Hand Raised:: Count: '+str(handRaisedCount))

                        components.html("""
                            <script>
                            const checkBxText = window.parent.document.querySelectorAll('.css-1djdyxw')
                            checkBxText[1].style.border = '1px solid #f94144'
                            </script>
                            """)
                    elif (not participateCbx) and chkClicked:
                        chkClicked = False

                        st.session_state.raisedCnt = handRaisedCount
                        st.session_state.chkStClicked = chkClicked

                        checked.write('Hand Lowered:: Count: '+str(handRaisedCount))

                        components.html("""
                            <script>
                            const checkBxText = window.parent.document.querySelectorAll('.css-1djdyxw')
                            checkBxText[1].style.border = '1px solid black'
                            </script>
                            """)
                    
                    chkClicked = st.session_state.chkStClicked
                    handRaisedCount = st.session_state.raisedCnt
                    st.session_state.raisedCnt = handRaisedCount

                    engagement_level = st.session_state.engagement_level
                    st.session_state.engagement_level = engagement_level
                    
                # emotion_count = st.session_state.emotion_count
                # print("End:",emotion_count)
                # st.session_state.emotion_count = {'Focused': 0, 'Distracted': 0}
                handRaisedCount = st.session_state.raisedCnt
                engagement_level = st.session_state.engagement_level
                print("End:",engagement_level,"\nHand Raised Count", handRaisedCount)
                st.session_state.engagement_level = 0
                st.session_state.raisedCnt = 0

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
            st.error("Incorrect Username/Password")
            # st.session_state["sidebarLoginCbx"] = False


if __name__ == '__main__':
    main()