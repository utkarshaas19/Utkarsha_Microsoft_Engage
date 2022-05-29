# Developer : Utkarsha Avirat Sutar
# Date : 17-May-2022
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
# Data required for training system to display on the terminal - UTKARSHA
utkINIT_LR = 1e-4
utkEPOCHS = 20
BS = 32
# To retrieve the image data from the respective folders with_mask and without_mask - UTKARSHA
utkDIRECTORY = r"C:\Users\LENOVO\Desktop\Engage\dataset"
CATEGORIES = ["with_mask", "without_mask"]
# Retrieve list of images from provided dataset and assign value - UTKARSHA
print("Images are getting loaded to the System")
data = []
labels = []
for category in CATEGORIES:
    projectpath = os.path.join(utkDIRECTORY, category)
    for img in os.listdir(projectpath):
    	img_path = os.path.join(projectpath, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)
    	data.append(image)
    	labels.append(category)
# Convert Text to Binary - UTKARSHA 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
# To Construct training image generator for data augmentation - UTKARSHA
aug = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
# Load the MobileNetV2 network
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
# Create head and the base model - UTKARSHA
utkheadModel = baseModel.output
utkheadModel = AveragePooling2D(pool_size=(7, 7))(utkheadModel)
utkheadModel = Flatten(name="flatten")(utkheadModel)
utkheadModel = Dense(128, activation="relu")(utkheadModel)
utkheadModel = Dropout(0.5)(utkheadModel)
utkheadModel = Dense(2, activation="softmax")(utkheadModel)
# Call head and the base model - UTKARSHA
model = Model(inputs=baseModel.input, outputs=utkheadModel)
# Loop over all layers in the base model and freeze them - UTKARSHA
for layer in baseModel.layers:
	layer.trainable = False
# Compiling Model - UTKARSHA
print("Compilation of the MODEL is going on...")
option = Adam(lr=utkINIT_LR, decay=utkINIT_LR / utkEPOCHS)
model.compile(loss="binary_crossentropy", optimizer=option,
	metrics=["accuracy"])
# Train the head of the network - UTKARSHA
print("Training Head Started...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=utkEPOCHS)
# Make predictions on the testing set - UTKARSHA
print("Network evaluation...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
print("saving mask model...")
model.save("mask_detector.model", save_format="h5")
# Plot the Training loss and accuracy - UTKARSHA
N = utkEPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")