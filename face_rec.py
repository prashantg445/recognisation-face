# Collect face data
import numpy as np
import cv2
import os
n_frames = 10  #no. of frames of your faces that the model will learn...more is good.
name = raw_input('\nWhat can i call you?\n')  # to remember user's name
#cam = cv2.VideoCapture(1)  # for camera other than default webcam like DroidCam...
cam = cv2.VideoCapture(0)   #default webcam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   #to detect faces

#if u want to change n_frames , change this also in face_recog.py
data = []   #array to store info. of your face
ix = 0   

while True:
	ret, frame = cam.read()   # ret will become false if system can't allocate any camera.
	fr = cv2.flip(frame,1)   # for filpping frame,i.e., to care for lateral inversion.
	if ret == True:
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)  # for converting in grayscale views  
		#since they require less processing and we want only face location so it is sufficient to achieve required accuracy
		faces = face_detector.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			fc = fr[y:y+h, x:x+w, :]  # original rgb frames(fr) are used for further processing

			r = cv2.resize(fc, (50, 50))  #changing the resolution of image so that less processing is required

			if ix%10 == 0 and len(data)<n_frames:  #capture every 10th frame , bcoz if consecutive frames are stored, they all will be similar so model will learn only a particular orientation of face 
				data.append(r)

			cv2.rectangle(fr, (x, y), (x+w, y+h), (0, 0, 255), 2) #to show rectangle on capturing faces.
		ix += 1
		cv2.imshow('Remembering your face', fr)
		if cv2.waitKey(1) == 27 or len(data) >= n_frames:
			break
	else:
		print ("error") #if no camera is available ,then print error
		break

cv2.destroyAllWindows()
data = np.asarray(data)

print (data.shape)
os.chdir(os.path.join(os.getcwd() ,'all_faces/'))  #all faces are stored in 'all_faces' folder.
np.save(name, data)