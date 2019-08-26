import cv2 
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg")
gray = cv2.imread("sample.jpg",cv2.IMREAD_GRAYSCALE)
def show_using_matplotlib():
	newImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	plt.imshow(newImg)
# We use cv2.cvtColor method if we use matplotlib to open the image to convert "BGR" values to "RGB" values


def show_using_cv2(img):
	cv2.imshow('cartoons image',img)
	cv2.waitKey(25000) # cv2.wait(0) will hold the image for infinite time 
					  # 2500 is the time in milliseconds
	cv2.destroyAllWindows()


show_using_cv2(gray)

