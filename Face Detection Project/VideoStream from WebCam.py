# This code contains the method to get video stream (frame by frame) from the webcam and
# draw a rectangle around the faces of the people

import cv2

cap = cv2.VideoCapture(0) # 0 refers to default webcam
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
	ret,frame = cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if ret == False: #Until ret is true video is not captured so we have to continue without going ahead
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	#scaling is done bcz we want that kernels work on the similar size images
	# 5 is no. of faces it can detect
	# this function will return the tuple of (x,y,w,h)
	# where x,y are the coordinates and w,h are the width and height


	

	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
		#(255,255,0) specifies the color
		# 2 specifies the thickness of the rectangle

	cv2.imshow("Video Frame",frame)

	#Wait for user input - q, then you will stop the loop
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()