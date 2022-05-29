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
            label = "MASK" 
            speak("You didn't wear a mask") 
            color = (0, 0, 255)
        # Display string result in Label - UTKARSHA 
        cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 5)
        cv2.imshow("Utkarsha's Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        exit()
cv2.destroyAllWindows()
vs.stop()   