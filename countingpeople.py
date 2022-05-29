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
        exit()
utkcapture.release()