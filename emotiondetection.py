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
        exit()
video.release()
cv2.destroyAllWindows()