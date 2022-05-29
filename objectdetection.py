# Developer : Utkarsha Avirat Sutar
# Date : 25-May-2022
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
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
# Ask for Arguments from user inorder to use the model and prototxt file for object detection - UTKARSHA
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# List of Objects that can be detected that are caught into the frame - UTKARSHA
Objects = ["KNIFE","AEROPLANE", "BICYCLE", "BIRD", "BOAT", "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DINING TABLE", "DOG", "HORSE", "BIKE", "HUMAN", "PLANT", "SHEEP", "SOFA SET", "TRAIN", "ELECTRONIC GADGET"]
COLORS = np.random.uniform(0, 255, size=(len(Objects), 3))
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# Start the Video Streaming - UTKARSHA
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
while True:
	# Frame Creation
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()
	for i in np.arange(0, detections.shape[2]):
		# Evaluate detections
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			index = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			label = "{}".format(Objects[index], confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[index], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[index], 2)
			speak(label+" is found")
	# Show Frame along with Name
	cv2.imshow("Utkarsha's Object Detector", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		exit()
	fps.update()
fps.stop()
cv2.destroyAllWindows()
vs.stop()