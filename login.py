import streamlit as st
import streamlit_authenticator as stauth
import subprocess

import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def main():
    """Login App"""

    st.title("LogIn")
    st.subheader("Login Section")

    username = st.text_input("User Name")
    password = st.text_input("Password",type='password')
    if st.button("Login"):
        create_usertable()
        hashed_pswd = make_hashes(password)

        result = login_user(username,check_hashes(password,hashed_pswd))
        if True:
        # if result:

            subprocess.Popen(["streamlit", "run", "app.py"])


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