import cv2
import os

# initalize the classifier
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml" # getting path to pre-trained models
faceCascade = cv2.CascadeClassifier(cascPath)

# apply faceCascade on webcam frames
video_capture = cv2.VideoCapture(0)

def detect_faces(frames):
    # input image is be greyscale for more accurate facial detection
    grey = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # detect faces in image. The faces variable contains a list of rectangular coordinates for every detected face (per frame)
    faces = faceCascade.detectMultiScale(
        grey,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE)
    
    # iterate over each face and draw a rectangle around each
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x,y), (x+w, y+h), (0,255,0), 2)
    
    return frames

################ VIDEO LOOP ################
while True:
    # capture frame-by-frame
    ret, frames = video_capture.read()

    # for each frame, try to detect a face
    detect_faces(frames)

    # display the resulting frame
    cv2.imshow('Video', frames)

    # if the user presses 'q', break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # goal: print name if face is recognized, else prompt user to input name (for future recognition)


#release the capture frames when loop is broken
video_capture.release()
cv2.destroyAllWindows()