import cv2
import numpy as np 

img = cv2.imread('Jamie_Before.jpg')

#img = cv2.resize(img,(500,500))

eyes_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
eyes = eyes_cascade.detectMultiScale(img,1.3,5)
print(eyes[0])
#print(eyes[1])
print(len(eyes))
x,y,w,h = eyes[0]

#for eye in eyes:
#	x,y,w,h = eye
#	frame = cv2.rectangle(img,(x,y),(x+h,y+w),(255,255,0),1)

	
print(x,y,h,w)
glasses = cv2.imread('glasses.png',cv2.IMREAD_UNCHANGED)
#glasses = cv2.resize(glasses,(h,w))



for c in range(3):
	for a in range(glasses.shape[0]):
		for b in range(glasses.shape[1]):
			#if((x+b-60)<x+h and (y+a-20)<y+w):
				if glasses[a][b][3]!=0:
					img[y+a-20][x+b-60][c] = glasses[a][b][c]



cv2.imshow('Jamie_Before',img)
#cv2.imshow('Jamie_after',glasses)
#img2 = cv2.addWeighted(img, 0.3, glasses, 0.7, 0) 
#cv2.imshow("final",img2)

print(glasses.shape)
print(img.shape)

#cv2.imshow('croppes',image)
cv2.waitKey(0)
