import numpy as np
import cv2 as cv
import joblib

# With these I just set some values for the HOG descriptor so it knows how to process the ROI I give it based on the red detection
#if you wish to know more about what each parameter does, check HOGDESscriptor's documentation. In short, it just sets the size
#of the chunks in which the ROI is processed.
winSize = (128, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True

hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                        L2HysThreshold, gammaCorrection, nlevels, signedGradients)
#------------ THE HOG DESCRIPTOR VALUES HAVE BEEN SET

#This is used  to detect all the red in the image and it returns the MASK (i.e. the RED is 1 and everything else is 0).
def detectRed(img):
    #This code is more or less off a neat tutorial. Mostly used that since I wasn't certain what range to pick for red.
    #If this doesn't dettect all ranges, I can just modify the values so it takes more "colors" into account, since that will just
    #decrease the performance a bit, but it will most likely not matter whatsoever.
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Range for lower red #Might as well let someone else pick the values
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv.inRange(imgHSV, lower_red, upper_red)
    # Range for upper range
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv.inRange(imgHSV, lower_red, upper_red)
    return (mask1+mask2)

#Here I use the "findContours" function to find the countours of the MASK I get from detecting RED. Then I follow this up by
#calculating its "moments" (https://www.quora.com/What-exactly-are-moments-in-OpenCV for an explanation of what they are).
#With said moments we calculate were the ROI should roughly be and then, using the SVM with HOG we test it.
def getCenters(img):
    finalMask = detectRed(img)
    kernel = np.ones((5, 5), np.uint8)
    finalMask = cv.morphologyEx(finalMask, cv.MORPH_OPEN, kernel)
    contours, hierarchy = cv.findContours(finalMask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    mu = [None] * len(contours)
    for i in range(len(contours)):
        mu[i] = cv.moments(contours[i])

    # Get the mass centers
    mc = [None] * len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

    print(len(contours))

    return mc

#Checking to see if the X coordinate is not out of image's bounds.
def withinBoundsX(coord, img):
    if(coord < 0 or coord > img.shape[1]):
        return 0
    return 1
#Analogous with the above, but for Y.
def withinBoundsY(coord, img):
    if(coord < 0 or coord > img.shape[0]):
        return 0
    return 1

#This is where the magic happens. We declare each points based on the previously found moments (we found the center with said moments).
#Since the HoG uses 128x128 images, we make it so the center remains in the center by adding/subtracting 64 to the right coordinates.
def detectSign(img, watch, mc):
    for points in mc:
        x1 = int(points[0]) - 64
        y1 = int(points[1]) - 64
        x2 = int(points[0]) + 64
        y2 = int(points[1]) + 64

        #This is a bit rough. Through trial and error I figured that there's a chance the ROI is just a bit below/above the
        #correct position, so I make 2 other ROI's, once 16 pixels below and one 16 pixels above. So we check 3 times for each
        #detected countour center.
        for disp in range(-16, 16+1, 16):
            if (not withinBoundsX(x1, img) or  not withinBoundsY(y1+disp, img) or not withinBoundsX(x2, img) or not withinBoundsY(y2+disp, img)):
                break
            #The window to check using the HOG.
            roi = img[y1+disp:y2+disp, x1:x2]

            #Computing the HOG and making it so we can give it to the SVM.
            descriptor = hog.compute(roi)
            descriptor = np.array(descriptor)
            descriptor = descriptor.transpose()
            #If the SVM gives a positive response (i.e. it is a stop sign) we show the image and draw a rectangle around the sign.
            if clf.predict(descriptor) == 1:
                cv.rectangle(watch, (x1, y1+disp), (x2, y2+disp), (255, 0, 0), 2)
                cv.imshow("pain", watch)
                cv.waitKey(0)

#I load the SVM that I've trained in a separate piece of code. This has been made using 30 positives and roughly 100 negatives.
#It is a linear SVM that can only classify whether something's a STOP SIGN or NOT.
#A simple way to imagine it is as such: you have two separate data points clusters, the SVM draws a line between the two
#thus dividing them into two categories. Whenever a new datapoint is added, it is given a category based on its position.
#Note: This is horribly undersimplified.
clf = joblib.load("stopSignLSVM.joblib")

#----------------VIDEO TESTING WITH FUNCTIONS
cap = cv.VideoCapture('Set2/picam19-02-14-07-39n.h264')
#Detect RED -> calclulate HoG -> give to SVM
while(cap.isOpened()):
    #Reading from the camera.
    ret, watch = cap.read()
    #watch = cv.pyrDown(watch)
    #This I added because in some cases the stop sign is too much to the right so we get an image out of bounds error,
    #so I just made the image a bit longer. A rough solution, but it works.
    watch = cv.copyMakeBorder(watch, 0, 0, 0, 128, cv.BORDER_REPLICATE)
    #The image is scaled down so it detects the stop sign at the right time.
    watch = cv.pyrDown(watch)
    #Get the centers.
    mc = getCenters(watch)
    #Detect the sign.
    detectSign(watch, watch, mc)
    #Show the image.
    cv.imshow("pain", watch)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.waitKey(0)
cap.release()
cv.destroyAllWindows()