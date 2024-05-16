import cv2
import os

# initalize the classifier
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml" # getting path to pre-trained models
faceCascade = cv2.CascadeClassifier(cascPath)
