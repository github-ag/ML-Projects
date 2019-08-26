# Recognize Faces using some Classification algorithm - like Logistic, KNN, SVM etc.


# 1. Read a video stream using open cv
# 2. extract faces out of it.
# 3. load the training data (numpy arrays of all the persons)
	# x - values stored in the numpy arrays.
	# y - values we need to assign for each person
# 4. use knn to find the prediction of face(int)
# 5. map the predicted id to the name of the user.
# 6. Display the predictions on the screen - bounding box and name

import numpy as np
import cv2
import os

########################## KNN code   ######################
def dist(a,b):
    return np.sqrt(sum((a-b)**2)) #a,b can have many dimensions.

def knn(X,Y,queryPoint,k=5):
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(X[i],queryPoint)
        vals.append([d,Y[i]])
    
    vals = sorted(vals)
    vals = vals[:k]
    
    vals = np.array(vals)
    
    counts = np.unique(vals[:,1],return_counts=True)
    max_frequency_index = counts[1].argmax() #gives the index of the maximum value.
    
    prediction = counts[0][max_frequency_index]
    return prediction

###############################################################


face_data = [] # stores the data of all the faces
path = './data/'
labels = [] #store the label for every data
class_id = 0 #It is the label for every file in the dataset.
names = {} # Mapping btw id - name 


################### Data Preperation #############################
for fx in os.listdir(path):
	if fx.endswith('.npy'):


		names[class_id] = fx[:-4] # creates mapping with class_id and the name
		data_item = np.load(path+fx)
		face_data.append(data_item)
		#print("loaded" + fx)
		#Create labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0) #converting a list of lists into a single list 
face_labels = np.concatenate(labels,axis=0)

#print(face_dataset.shape)
#print(face_labels.shape)

############# Getting a video stream

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0

while True:
	ret,frame = cap.read()
	if ret==False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	for face in faces:
		x,y,w,h = face

		#Get the region of interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		out = knn(face_dataset,face_labels,face_section.flatten())
		#print(out)
		
		#Display on the screen name and rectangle around it
		predicted_name = names[int(out)]
		cv2.putText(frame,predicted_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
