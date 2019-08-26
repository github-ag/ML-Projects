############ COLLECTING THE FACE DATA OF THE PERSON  ############################################

# 1. Read a Video Stream from Camera (Frame by Frame)
# 2. Detect faces and show bounding box
# 3. Flatten the largest face image(gray scale image) and save in numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

# Iniatializing the camera
cap = cv2.VideoCapture(0)

# For face detection in the video we need haar cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


skip = 0
face_data = []
path = './data/'
file_name = "Abhishek"


while True:
	ret,frame = cap.read()

	if ret == False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	# Now sort the face to get the largest face
	faces = sorted(faces,key = lambda f:f[2]*f[3])


	#Creating rectangle around the largest face
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

		#Extract(Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		# Store every 10th face
		if(skip%10 == 0):
			face_data.append(face_section)
			print(len(face_data)) #face data is the matrix of RGB value of every pixel
		skip+=1

		cv2.imshow("Frame",frame)
		cv2.imshow("face Section",face_section)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save this data into the file system
np.save(path+file_name+'.npy',face_data)
print("Data Succesfully Saved at "+path+file_name)


cap.release()
cv2.destroyAllWindows()