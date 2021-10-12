# To run the script :
# 1) Install streamlit :
#                       a) For Windows : pip install streamlit
# 2) Test it using streamlit hello : This opens up a demo app.
# 3) Type in cmd : streamlit run Model.py 

# Importing the required libraries
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from PIL import Image
import face_recognition 

# To change the cotents of the screen
placeholder=st.empty()

# The default screen
with placeholder.beta_container():
        st.title("Computer Vision Operations")
        st.subheader("Computer Vision Operations On Images To Perform Different Tasks.")
        st.write("This application is made by Harshavardhan Bajoria. It performs various operations witih OpenCV")

# The selection bar
add_selectbox = st.sidebar.selectbox(
    "What Operation You Would Like To Perform :",
    ("About","GrayScale", "Change RGB Value", "Face Mesh","Change Background","Face Detection")
)

#The About Section
if add_selectbox=="About":
    st.write()
    st.write("This App Uses OpenCV library, mediapipe library and streamlite library to perform the different tasks.")
    st.write("On the right hand side you may choose the tasks you wish to perform.")
    st.write("Currently it supports only images the live webcam feature wil be added soon.")
    st.write("Thanks For Visting. Hope You Have A Great Day. ")
    st.write()
    st.subheader("You May Report Bugs By Using Any Contact Method Listed Below.")
    st.subheader("Email :")
    st.write("HVBAJORIA@hotmail.com")
    st.subheader("LinkedIn:")
    st.write("https://www.linkedin.com/in/harshavardhan-bajoria")

# To get the image from the user
image=st.sidebar.file_uploader("Upload Image")
if image is not None:
    image=Image.open(image)
    image=np.array(image)
    st.sidebar.image(image)

# The GrayScale Section
    if add_selectbox=="GrayScale":
        placeholder.empty()
        with placeholder.beta_container():
              st.title("GrayScaled Image :")
              st.write("This application is made by Harshavardhan Bajoria. It performs various operations witih OpenCV")
                #Makes the image grayscaled
              image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
              st.image(image)

# The RGB Value Section
    elif add_selectbox=="Change RGB Value":
        placeholder.empty()
        with placeholder.beta_container():
            st.title("Change RGB Values :")
            st.write("This application is made by Harshavardhan Bajoria. It performs various operations witih OpenCV")
            # To get value from the user. By default the original image is shown.
            red=st.slider("Red",0,255,None,1)
            green=st.slider("Green",0,255,None,1)
            blue=st.slider("Blue",0,255,None,1)
            zeros=np.zeros((image.shape[0],image.shape[1]),np.uint8)
            b,g,r=cv2.split(image)
                # To get the final image with new RGB values
            custom=cv2.merge([b+blue,g+green,r+red])
            st.image(custom)

# The Face Mesh Section
    elif add_selectbox=="Face Mesh":
        placeholder.empty()
        with placeholder.beta_container():
            st.title("Face Mesh :")
            st.write("This application is made by Harshavardhan Bajoria. It performs various operations witih OpenCV")
            mp_face_detect=mp.solutions.face_detection
            model_detection=mp_face_detect.FaceDetection()
            mp_drawing=mp.solutions.drawing_utils
            # To draw the mesh
            drawing_spec = mp_drawing.DrawingSpec((255,0,255),thickness=2, circle_radius=1)
            mp_face_mesh=mp.solutions.face_mesh
            model_facemesh=mp_face_mesh.FaceMesh()
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable= False
            results= model_facemesh.process(image)
            image.flags.writeable= True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Drawimg The Mesh
            if results.multi_face_landmarks:
                for landmark in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image=image,landmark_list=landmark,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
            st.image(image)

# The Change Background Section
    elif add_selectbox=="Change Background":
        placeholder.empty()
        with placeholder.beta_container():
            st.title("Background Changer")
            st.write("This application is made by Harshavardhan Bajoria. It performs various operations witih OpenCV")
            # The selction box
            add_selectbox2 = st.selectbox(
               "Choose A Background:",
               ("Light Room", "Desert", "Beach","Wallpaper","Earth","Mountains")
            )
            mp_drawing=mp.solutions.drawing_utils
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            if add_selectbox2=="Light Room":
                BG_COLOR = cv2.imread('background.jpg')
                selfie_segmentation= mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
                bg_image = None
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = selfie_segmentation.process(image)
                image.flags.writeable = True
                BG_COLOR=cv2.resize(BG_COLOR,(image.shape[1],image.shape[0]))
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.773
                if bg_image is None:
                     bg_image = np.zeros(image.shape, dtype=np.uint8)
                     bg_image[:] = BG_COLOR
                output_image = np.where(condition, image, bg_image)
                output_image=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
                st.image(output_image)
            if add_selectbox2=="Desert":
                BG_COLOR = cv2.imread('background2.jpg')
                selfie_segmentation= mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
                bg_image = None
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = selfie_segmentation.process(image)
                image.flags.writeable = True
                BG_COLOR=cv2.resize(BG_COLOR,(image.shape[1],image.shape[0]))
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.773
                if bg_image is None:
                     bg_image = np.zeros(image.shape, dtype=np.uint8)
                     bg_image[:] = BG_COLOR
                output_image = np.where(condition, image, bg_image)
                output_image=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
                st.image(output_image)
            if add_selectbox2=="Beach":
                BG_COLOR = cv2.imread('background3.jpg')
                selfie_segmentation= mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
                bg_image = None
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = selfie_segmentation.process(image)
                image.flags.writeable = True
                BG_COLOR=cv2.resize(BG_COLOR,(image.shape[1],image.shape[0]))
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.773
                if bg_image is None:
                     bg_image = np.zeros(image.shape, dtype=np.uint8)
                     bg_image[:] = BG_COLOR
                output_image = np.where(condition, image, bg_image)
                output_image=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
                st.image(output_image)
            if add_selectbox2=="Wallpaper":
                BG_COLOR = cv2.imread('background4.jpg')
                selfie_segmentation= mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
                bg_image = None
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = selfie_segmentation.process(image)
                image.flags.writeable = True
                BG_COLOR=cv2.resize(BG_COLOR,(image.shape[1],image.shape[0]))
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.773
                if bg_image is None:
                     bg_image = np.zeros(image.shape, dtype=np.uint8)
                     bg_image[:] = BG_COLOR
                output_image = np.where(condition, image, bg_image)
                output_image=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
                st.image(output_image)
            if add_selectbox2=="Earth":
                BG_COLOR = cv2.imread('background5.jpg')
                selfie_segmentation= mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
                bg_image = None
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = selfie_segmentation.process(image)
                image.flags.writeable = True
                BG_COLOR=cv2.resize(BG_COLOR,(image.shape[1],image.shape[0]))
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.773
                if bg_image is None:
                     bg_image = np.zeros(image.shape, dtype=np.uint8)
                     bg_image[:] = BG_COLOR
                output_image = np.where(condition, image, bg_image)
                output_image=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
                st.image(output_image)
            if add_selectbox2=="Mountains":
                BG_COLOR = cv2.imread('background6.jpg')
                selfie_segmentation= mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
                bg_image = None
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = selfie_segmentation.process(image)
                image.flags.writeable = True
                BG_COLOR=cv2.resize(BG_COLOR,(image.shape[1],image.shape[0]))
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.773
                if bg_image is None:
                     bg_image = np.zeros(image.shape, dtype=np.uint8)
                     bg_image[:] = BG_COLOR
                output_image = np.where(condition, image, bg_image)
                output_image=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
                st.image(output_image)

# The Face Detection Section
    elif add_selectbox=="Face Detection":
        placeholder.empty()
        with placeholder.beta_container():
            st.title("Face Detection :")
            st.write("This application is made by Harshavardhan Bajoria. It performs various operations witih OpenCV")
            mp_drawing=mp.solutions.drawing_utils
            mp_face_detect=mp.solutions.face_detection
            model_detection=mp_face_detect.FaceDetection()
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable= False
            results= model_detection.process(image)
            image.flags.writeable= True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # To draw the points and boundary
            if results.detections :
                for landmark in results.detections:
                    mp_drawing.draw_detection(image,landmark)
            st.image(image)
