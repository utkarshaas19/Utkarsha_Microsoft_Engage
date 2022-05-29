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
        exit()
utkcapture.release()