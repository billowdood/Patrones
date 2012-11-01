import numpy as np
import cv2

#Program which intend to recognize digit patterns,opencv 2.4.2 was used#

#Training!
#For the training,the image of the digit is displayed and it is expected that the user press on his keyboard the digit is shown on screen
def training(contoursObj):
	#Empty array for the labels
	samples =  np.empty((0,100))
	#Empty array for the corresponding value
	valueDigit = []
	#Loop over all the images,in this case we have 9 images per digit,so in total we have 90 images(we included 0 as a digit)
	for num_image in range(1,91):
		#Get the image and find the contours of the number(rectangle)
		image_digit = cv2.imread("Samples\\"+str(num_image)+".jpg")
		#Magic thingies to te image so the contours can be more precise
		gray = cv2.cvtColor(image_digit,cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(5,5),0)
		thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
		#Coordinates of the rectangle (x,y)->starting points, (w,h)-> width and height
		x,y,w,h = contours(image_digit)
		cv2.imshow('digit',image_digit)
		roi = thresh[y:y+h,x:x+w]
		roismall = cv2.resize(roi,(10,10))
		key = cv2.waitKey(0)
		#Assign the value of the image
		valueDigit.append(int(chr(key)))
		sample = roismall.reshape((1,100))
		#Add the label to the array
   		samples = np.append(samples,sample,0)
	#Array of the images
	valueDigit = np.array(valueDigit,np.float32)
	valueDigit = valueDigit.reshape((valueDigit.size,1))
	print "training complete"
	np.savetxt('generalsamples.data',samples)
	np.savetxt('generalvaluedigits.data',valueDigit)

#Function for finding contours of the image which is given as a parameter
def contours(imageToDraw):
	imgray = cv2.cvtColor(imageToDraw,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,127,255,0)
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	x,y,w,h = cv2.boundingRect(contours[0])
	return x,y,w,h


#Testing!
def loadSamples():
	#Load the patterns and the labels
	samples = np.loadtxt('generalsamples.data',np.float32)
	responses = np.loadtxt('generalvaluedigits.data',np.float32)
	responses = responses.reshape((responses.size,1))
	#We use the opencv built-in function KNeartest
	model = cv2.KNearest()
	#We train the algorithm
	model.train(samples,responses)
	return model

#Function for testing the algorithm
def testAlgorithm(algorithmKNearest):
	#Loop through all the images
	for image_testing in range(1,91):
		#Read image for testing
		imTest = cv2.imread('TestingImages\\'+str(image_testing)+'.jpg')
		#Output image
		out = np.zeros(imTest.shape,np.uint8)
		gray = cv2.cvtColor(imTest,cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
		#Find the contours of the image
		contors,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		#Bounding rectangle of the digit
		x,y,w,h = contours(imTest)
		#Resize the rectangle and store just some pixels
		roi = thresh[y:y+h,x:x+w]
		roismall = cv2.resize(roi,(10,10))
		roismall = roismall.reshape((1,100))
		roismall = np.float32(roismall)
		#We use the KNearest function find_nearest with k = 1
		retval,results,neigh_resp,dists = algorithmKNearest.find_nearest(roismall,k = 1)
		string = str(int((results[0][0])))
		cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
		#Show te original image and the result
		cv2.imshow('imTest',imTest)
		cv2.imshow('imOut',out)
		cv2.waitKey(0)

#Main function
def main():
	training(contours)
	kNearestNeighbourhood = loadSamples()
	testAlgorithm(kNearestNeighbourhood)

main()