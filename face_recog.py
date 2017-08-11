# Face recognition
import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture(1) #for Droidcam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
font = cv2.FONT_HERSHEY_SIMPLEX
# Load data in train (X and y)
n_frames=10
names=[]
os.chdir(os.path.join(os.getcwd(),'all_faces/'))  #to collect all names
data = np.asarray([], dtype = 'int8')  # since pixels lie b/w 0-255 ,hence 'int8'
for each in os.listdir(os.getcwd()):
	names.append(each[:-4])   # -4 for avoiding extension '.npy' attached wit each file and only name is retrieved
	face = np.load(each).reshape((n_frames, -1))
	#preparing data for model
	if(data.shape[0]==0):
		data=face
	else:
		data = np.vstack((data,face))
#preparing labels
labels =  np.zeros((data.shape[0],))
for ix in range(labels.shape[0]/n_frames):
	labels[ix*n_frames : (ix+1)*n_frames] = ix

# Define KNN functions manually from scratch 
def distance(x1, x2):
	d = np.sqrt(((x1-x2)**2).sum())  # Euclidean distance to evaluate similarity. 
	return d

def knn(X_train, y_train, xt, k=5):  #xt is testing input 
	vals = []
	for ix in range(X_train.shape[0]):
		d = distance(X_train[ix], xt) 
		vals.append([d, y_train[ix]])  #calculating similarity of testing input and training data
	sorted_labels = sorted(vals, key=lambda z: z[0]) #elements sorted on basis of similarity
	neighbours = np.asarray(sorted_labels)[:k, -1]   #k nearest neighbours are selected.
	
	freq = np.unique(neighbours, return_counts=True) #frequency of similar elements are recorded
	
	return freq[0][freq[1].argmax()]  # returned the most frequent similar element


#knn function from sklearn library

knn2 = KNeighborsClassifier()  
knn2.fit(data,labels)

# Run the main loop
while True:
	ret, frame = cam.read()
	fr = cv2.flip(frame,1)
	if ret == True:
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		faces = face_detector.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			# Extract detected face
			fc = fr[y:y+h, x:x+w, :]
			# resize to a fixed shape
			r = cv2.resize(fc, (50, 50)).flatten()

			text = names[int(knn2.predict(r))]  # for scratch,knn(data, labels, r) inside int()
			cv2.putText(fr, text, (x, y), font, 1, (255, 0, 0), 2)  #for showing name of recognized person

			cv2.rectangle(fr, (x, y), (x+w, y+h), (0, 0, 255), 2)
		cv2.imshow('hello', fr)
		if cv2.waitKey(1) == 27:
			break
	else:
		print "error"
		break

cv2.destroyAllWindows()
print(data.shape)
