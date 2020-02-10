import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import os

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
X = []
Y = []

def readFile(name, kind):
    for file_type in [name]:
        for imageName in os.listdir(file_type):
            path = str(file_type) + "/" + str(imageName)
            img = cv.imread(path)
            descriptor = hog.compute(img)
            descriptor = np.array(descriptor)
            descriptor = descriptor.transpose()
            print(len(descriptor[0]))
            print("-------------")
            X.append(descriptor[0])
            Y.append(kind)

def detectRed(img):
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

def withinBoundsX(coord, img):
    if(coord < 0 or coord > img.shape[1]):
        return 0
    return 1

def withinBoundsY(coord, img):
    if(coord < 0 or coord > img.shape[0]):
        return 0
    return 1

def detectSign(img, watch, mc):
    for points in mc:
        x1 = int(points[0]) - 64
        y1 = int(points[1]) - 64
        x2 = int(points[0]) + 64
        y2 = int(points[1]) + 64
        #Check if it's within bounds
        #watch = cv.pyrDown(watch)
        #watch = cv.copyMakeBorder(watch, 0, 0, 0, 64, cv.BORDER_CONSTANT)
        #cv.imshow("pain", watch)
        #cv.rectangle(watch, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for disp in range(-16, 16+1, 16):
            if (not withinBoundsX(x1, img) or  not withinBoundsY(y1+disp, img) or not withinBoundsX(x2, img) or not withinBoundsY(y2+disp, img)):
                break
            roi = img[y1+disp:y2+disp, x1:x2]
            #cv.imshow("roi", roi)
            descriptor = hog.compute(roi)
            descriptor = np.array(descriptor)
            descriptor = descriptor.transpose()
            if clf.predict(descriptor) == 1:
                cv.rectangle(watch, (x1, y1+disp), (x2, y2+disp), (255, 0, 0), 2)
                cv.imshow("pain", watch)
                cv.waitKey(0)
                # cv.imwrite("mining/mined" + str(index) + ".png", roi)

readFile("signs", 1)
readFile("negatives", -1)
X = np.array(X)
Y = np.array(Y)
print(X)
print(Y)
X, Y = shuffle(X, Y)
clf = svm.SVC(kernel = "linear", gamma = "auto")
clf.fit(X, Y) #USING THE WHOLE DATASET
#------------------THE SVM HAS BEEN TRAINED

#------------------Reading image
mining = cv.imread("test24.png")
watch = cv.imread("test24.png")
modify = cv.imread("test24.png")

#------------------COUNTOUR PLAY
'''detectRed(watch)
mc = getCenters(watch)
print(len(mc))
for points in mc:
    x1 = int(points[0]) - 64
    y1 = int(points[1]) - 64
    x2 = int(points[0]) + 64
    y2 = int(points[1]) + 64
    # Check if it's within bounds
    # watch = cv.pyrDown(watch)
    #watch = cv.copyMakeBorder(watch, 0, 0, 0, 64, cv.BORDER_REPLICATE)
    # cv.imshow("pain", watch)
    cv.rectangle(watch, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for disp in range(-16, 16+1, 16):
        if (not withinBoundsX(x1, watch) or not withinBoundsY(y1+disp, watch) or not withinBoundsX(x2, watch) or not withinBoundsY(y2+disp, watch)):
            break

        roi = watch[y1+disp:y2+disp, x1:x2]
        cv.imshow("roi", roi)
        print(disp)
        #cv.rectangle(watch, (x1, y1+disp), (x2, y2+disp), (0, 0, 255), 2)
        # cv.imshow("roi", roi)
        descriptor = hog.compute(roi)
        descriptor = np.array(descriptor)
        descriptor = descriptor.transpose()
        if clf.predict(descriptor) == 1:
            cv.rectangle(watch, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.imshow("pain", watch)
            cv.waitKey(0)'''
#cv.imshow("watch", watch)
#---------------COUNTOUR PLAY UP!!! ^ ^ ^

#detectSign(watch, watch, mc)

#----------------VIDEO TESTING WITH FUNCTIONS
#----------------------VIDEO TESTING BEGINS
cap = cv.VideoCapture('Set2/picam19-02-14-07-39n.h264')
#Detect RED -> calclulate HoG -> give to SVM
while(cap.isOpened()):
    ret, watch = cap.read()
    #watch = cv.pyrDown(watch)
    watch = cv.copyMakeBorder(watch, 0, 0, 0, 128, cv.BORDER_REPLICATE)
    mc = getCenters(watch)
    detectSign(watch, watch, mc)
    cv.imshow("pain", watch)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


#-------------------THIS IS JUST MINING
'''index = 99
for y in range(1, mining.shape[0] - 128, 10):
    for x in range(1, mining.shape[1] - 128, 10):
        roi = mining[y:y+128, x:x+128]
        descriptor = hog.compute(roi)
        descriptor = np.array(descriptor)
        descriptor = descriptor.transpose()
        if clf.predict(descriptor) == 1:
            cv.rectangle(watch, (x, y), (x + 128, y + 128), (255, 0, 0), 2)
            cv.imshow("pain", watch)
            cv.waitKey(10)
            #cv.imwrite("mining/mined" + str(index) + ".png", roi)
            index+=1
#cv.imshow("pain", testImg)
cv.waitKey(0)'''
#---------------------MINING ENDS HERE, DID FUCKING WONDERS

#----------------------VIDEO TESTING BEGINS
'''cap = cv.VideoCapture('Set2/picam19-02-14-07-39n.h264')
#Detect RED -> calclulate HoG -> give to SVM
while(cap.isOpened()):
    ret, testImg= cap.read()
    watch = testImg
    for y in range(1,  int(testImg.shape[0]/2) - 128 - 128, 64):
        for x in range(int(testImg.shape[1]/1.5), testImg.shape[1] - 128, 64):
            #cv.rectangle(testImg, (x, y), (x + 128, y + 128), (255, 0, 0), 2)
            roi = testImg[y:y+128, x:x+128]

            maskFin= detectRed(roi)
            whitePixels = cv.countNonZero(maskFin)
            cv.rectangle(watch, (x, y), (x + 128, y + 128), (255, 255, 0), 2)
            #print(whitePixels)
            #if whitePixels > 4000:
                #cv.imshow("White!", maskFin)
                #cv.waitKey(0)
            if 1:
                descriptor = hog.compute(roi)
                descriptor = np.array(descriptor)
                descriptor = descriptor.transpose()
                if clf.predict(descriptor) == 1:
                    cv.rectangle(testImg, (x, y), (x + 128, y + 128), (0, 255, 0), 2)
                    #cv.imshow("lol", roi)
                    cv.imshow("pain", watch)
                    cv.waitKey(1)
                else:
                    cv.imshow("pain", maskFin)
                    pass
            else:
                cv.imshow("pain", watch)
                pass
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.imshow("pain", testImg)'''
cv.waitKey(0)