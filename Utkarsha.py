from cProfile import label
from distutils.cmd import Command
from distutils.command import config
from re import I
from tkinter import *
import tkinter
from tkinter.ttk import *
from turtle import bgcolor, color
from tkvideo import tkvideo
import time
import cv2
from PIL import ImageTk,Image
def EngageProject():
    def handgestures():
        # Utkarsha's Hand Gestures Detector
        # Developer : Utkarsha Avirat Sutar
        # Date : 13-May-2022
        import cv2
        import numpy as np
        import mediapipe as mp
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        import pyttsx3
        engine=pyttsx3.init('sapi5')
        voices=engine.getProperty('voices')
        engine.setProperty('voice',voices[1].id)
        def speak(audio):
            engine.say(audio)
            engine.runAndWait()
        if __name__ == "__main__":
            pass
        speak("Welcome to the Hand Gestures Detection module. This module will help you recognize the hand gestures")
        # Initializing Mediapipe - UTKARSHA
        utkHands = mp.solutions.hands
        hands = utkHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        utkDraw = mp.solutions.drawing_utils
        # Getting Files - UTKARSHA
        utkHandmodel = load_model('hand_gestures')
        # We have a file with .names extension in which all the gesture names are specified - UTKARSHA
        f = open('gesture.names', 'r')
        utkgesturenm = f.read().split('\n')
        f.close()
        print(utkgesturenm)
        # Video Capturing - UTKARSHA
        utkcapture = cv2.VideoCapture(0)
        # Frame designing and Identification of hand gestures by plotting points - UTKARSHA
        while True:
            _, utkframe = utkcapture.read()
            x, y, c = utkframe.shape
            utkframe = cv2.flip(utkframe, 1)
            utkframergb = cv2.cvtColor(utkframe, cv2.COLOR_BGR2RGB)
            result = hands.process(utkframergb)
            className = ''
            if result.multi_hand_landmarks:
                landmarks = []
                for utkHandloop in result.multi_hand_landmarks:
                    for lm in utkHandloop.landmark:
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])
                    utkDraw.draw_landmarks(utkframe, utkHandloop, utkHands.HAND_CONNECTIONS)
                    prediction = utkHandmodel.predict([landmarks])
                    classID = np.argmax(prediction)
                    className = utkgesturenm[classID]
            speak(className)
            cv2.putText(utkframe, className, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Utkarsha's Hand Gestures Detector", utkframe) 
            # To Quit from the Application - UTKARSHA
            if cv2.waitKey(1) == ord('q'):
                utkframe.destroy()
        utkcapture.release()
    def maskdetection():
        # Utkarsha's Mask Detector
        # Developer : Utkarsha Avirat Sutar
        # Date : 21-May-2022
        import os
        import numpy as np
        import imutils
        import time
        import cv2
        from os.path import dirname, join
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from imutils.video import VideoStream
        import pyttsx3
        engine=pyttsx3.init('sapi5')
        voices=engine.getProperty('voices')
        engine.setProperty('voice',voices[1].id)
        def speak(audio):
            engine.say(audio)
            engine.runAndWait()
        if __name__ == "__main__":
            pass
        speak("Welcome to the mask detection module. COVID-19 has made it necessary to wear a mask. this module will help you find whether a person is wearing mask or not")
        def utkdetectmask(frame, faceNet, maskNet):
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
            faceNet.setInput(blob)
            detections = faceNet.forward()
            print(detections.shape)
            faces = []
            locations = []
            preds = []
            # Perform Detections on iterations - UTKARSHA
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    # Change its colour and also resize it - UTKARSHA
                    utkface = frame[startY:endY, startX:endX]
                    utkface = cv2.cvtColor(utkface, cv2.COLOR_BGR2RGB)
                    utkface = cv2.resize(utkface, (224, 224))
                    utkface = img_to_array(utkface)
                    utkface = preprocess_input(utkface)
                    # While detecting faces, build a box arround the face - UTKARSHA 
                    faces.append(utkface)
                    locations.append((startX, startY, endX, endY))
            # At least one face should be detected for predicting - UTKARSHA
            if len(faces) > 0:
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)
            return (locations, preds)
        # Load Model predict face - UTKARSHA
        utkprototxtPath = r"deploy.protext"
        utkweightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
        utkfaceNet = cv2.dnn.readNet(utkprototxtPath, utkweightsPath)
        # Load Model predict Mask - UTKARSHA
        maskNet = load_model("mask_detector.model")
        # initialize the video stream
        print("Starting Camera")
        vs = VideoStream(src=0).start()
        # Perform iterations for Video Streaming - UTKARSHA
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            (locations, preds) = utkdetectmask(frame, utkfaceNet, maskNet)
            for (box, pred) in zip(locations, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                # Check for conditions - UTKARSHA
                if(mask > withoutMask):
                    label = "MASK" 
                    speak("You Have wore a mask") 
                    color = (0, 255, 0)
                else:
                    label = "No MASK" 
                    speak("You didn't wear a mask") 
                    color = (0, 0, 255)
                # Display string result in Label - UTKARSHA 
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 5)
                cv2.imshow("Utkarsha's Mask Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                frame.destroy()
        vs.stop()   
    def counting():
        # Utkarsha's Counting People Application
        # Developer : Utkarsha Avirat Sutar
        # Date : 15-May-2022
        import time
        import numpy as np
        import cv2
        import pyttsx3
        engine=pyttsx3.init('sapi5')
        voices=engine.getProperty('voices')
        engine.setProperty('voice',voices[1].id)
        def speak(audio):
            engine.say(audio)
            engine.runAndWait()
        if __name__ == "__main__":
            pass
        speak("Welcome to the Counting People Model. This module will help you count number of people in a frame")
        time.sleep(0.1)
        utkhog = cv2.HOGDescriptor()
        utkhog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # Video Capturing - UTKARSHA
        utkcapture = cv2.VideoCapture(0)
        while True:
            ret, utkframe = utkcapture.read()
            utkflipped = cv2.flip(utkframe, flipCode = 1)
            utkframe1 = cv2.resize(utkflipped, (640, 480))
            # Required for displaying frame - UTKARSHA
            boxes, weights = utkhog.detectMultiScale(utkframe1, winStride=(8,8))
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])  
            for (xA, yA, xB, yB) in boxes:
                # Displays the detected figures in colour picture and else will be black and white - UTKARSHA
                cv2.rectangle(utkframe1, (xA, yA), (xB, yB),(0, 255, 0), 2)
                b=len(boxes)
                # Displays count of people after evaluating - UTKARSHA
                cv2.putText(utkframe1,"Number of People : "+str(b),(30,50),0,1,(255,255,255),2)
                speak("There are "+str(b)+" people in the frame") 
            img = cv2.resize(utkframe1,(640,480))
            cv2.imshow("Utkarsha's Counting Number of People Application", utkframe1)
            # To Quit from the Application - UTKARSHA
            if cv2.waitKey(1) == ord('q'):
                utkframe1.destroy()
        utkcapture.release()
    def emotiondetection():
        # Utkarsha's Emotion Detector
        # Developer : Utkarsha Avirat Sutar
        # Date : 22-May-2022
        import cv2
        import numpy as np
        from keras.models import load_model
        import pyttsx3
        engine=pyttsx3.init('sapi5')
        voices=engine.getProperty('voices')
        engine.setProperty('voice',voices[1].id)
        def speak(audio):
            engine.say(audio)
            engine.runAndWait()
        if __name__ == "__main__":
            pass
        speak("Welcome to the Emotion Detection Module.Emotions are very important and play a vital role in persons life. This module will help you to predict the emotions of Human beings")
        # Load model into program - UTKARSHA
        model=load_model('model_file_30epochs.h5')
        # Enable Capturing real time video - UTKARSHA
        video=cv2.VideoCapture(0)
        utkfaceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Emotions that can be detected - UTKARSHA
        emotions={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
        # Detect emotions until user exits the Application - UTKARSHA
        while True:
            ret,frame=video.read()
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces= utkfaceDetect.detectMultiScale(gray, 1.3, 3)
            for x,y,w,h in faces:
                sub_face_img=gray[y:y+h, x:x+w]
                resized=cv2.resize(sub_face_img,(48,48))
                normalize=resized/255.0
                reshaped=np.reshape(normalize, (1, 48, 48, 1))
                result=model.predict(reshaped)
                label=np.argmax(result, axis=1)[0]
                print(label)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
                cv2.putText(frame, emotions[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) 
                speak("Person is "+emotions[label])
            # Display Frame of project until user quits using 'q' - UTKARSHA
            cv2.imshow("Utkarsha's Emotion Detector",frame)
            k=cv2.waitKey(1)
            if k==ord('q'):
                frame.destroy()
        video.release()
    def objectdetection():
        # Utkarsha's Object Detector
        import pyttsx3
        engine=pyttsx3.init('sapi5')
        voices=engine.getProperty('voices')
        engine.setProperty('voice',voices[1].id)
        def speak(audio):
            engine.say(audio)
            engine.runAndWait()
        if __name__ == "__main__":
            pass
        speak("Welcome to the Object Detection Module. This module will help you identify and recognize objects in the real world")
    def quitapp():
        import pyttsx3
        engine=pyttsx3.init('sapi5')
        voices=engine.getProperty('voices')
        engine.setProperty('voice',voices[1].id)
        def speak(audio):
            engine.say(audio)
            engine.runAndWait()
        if __name__ == "__main__":
            pass
        speak("Thankyou, do visit our application again")
        exit()
    win1=Tk()
    win1.title('Avishkar - The Face Detection Application')
    win1.configure(background='black')
    win1.geometry("1200x700")
    bgimg= tkinter.PhotoImage(file = "backgroundimage.png")
    limg= Label(win1, i=bgimg, background='black')
    limg.place(x=0,y=300)
    bgimg1= tkinter.PhotoImage(file = "titleimage.png")
    limg1= Label(win1, i=bgimg1, background='black')
    limg1.place(x=0,y=0)
    b1=Button(win1, text="HAND GESTURES DETECTION", width=50, command=handgestures)
    b1.place(relx=0.5, rely=0.5, anchor=CENTER)
    b2=Button(win1, text="MASK DETECTION", width=50, command=maskdetection)
    b2.place(relx=0.5, rely=0.57, anchor=CENTER)
    b3=Button(win1, text="COUNTING NUMBER OF PEOPLE IN A FRAME", width=50, command=counting)
    b3.place(relx=0.5, rely=0.64, anchor=CENTER)
    b4=Button(win1, text="EMOTION DETECTION", width=50, command=emotiondetection)
    b4.place(relx=0.5, rely=0.71, anchor=CENTER)
    b5=Button(win1, text="OBJECT DETECTION", width=50, command=objectdetection)
    b5.place(relx=0.5, rely=0.78, anchor=CENTER)
    b6=Button(win1, text="QUIT", width=50, command=quitapp)
    b6.place(relx=0.5, rely=0.85, anchor=CENTER)
    win1.mainloop()
EngageProject()