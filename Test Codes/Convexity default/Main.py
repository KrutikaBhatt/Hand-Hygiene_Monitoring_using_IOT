# Using Convexity defect

"""
import cv2
import numpy as np

hand = cv2.imread("./TrainData/Greyscale2.jpg",0)

ret, img = cv2.threshold(hand,70,255,cv2.THRESH_BINARY)

countours,hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# A one-liner for loop across the contours
hull = [cv2.convexHull(c) for c in countours]
final = cv2.drawContours(hand, hull, -1, (255,255,255))
status = cv2.imwrite(r'./Output/ConvexHull1.jpg', final)
print(status)

cv2.imshow("Hand_Image",hand)
cv2.imshow("Threshold",img)
cv2.imshow("Convex_hull",final)


cv2.waitKey(0)
cv2.destroyAllWindows()

"""

import cv2
import numpy as np
import math
import time

#Start the webCam
cam = cv2.VideoCapture(0)

while cam.isOpened():
    #Read the frames
    var, frames = cam.read()

    #Get the hand Data from the rectangle grid
    cv2.rectangle(frames,(100,100),(400,400),(0,255,0),1)
    crop_image = frames[100:400,100:400]

    #Apply Gaussian blur -Remove noise
    blur = cv2.GaussianBlur(crop_image,(3,3),0)

    #Change from bgr->hsv
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    #Creating a mask - Check the color ranges using a trackbar
    mask = cv2.inRange(hsv,np.array([2,0,0]),np.array([20,255,255]))

    #Dilation and Erosion
    #Kernel
    kernel = np.ones((5,5))

    dilation = cv2.dilate(mask,kernel,iterations=1)
    erosion = cv2.erode(dilation,kernel,iterations=1)

    filtered = cv2.GaussianBlur(erosion,(3,3),0)
    ret,thresh = cv2.threshold(filtered,127,255,0)

    cv2.imshow("Thresholded",thresh)

    #Contouring
    countours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Find contours with max Area
    try:
        main_contour = max(countours,key = lambda x: cv2.contourArea(x) )

        x1,y1,x2,y2 = cv2.boundingRect(main_contour)
        cv2.rectangle(crop_image,(x1,y1),(x1+x2, y1+y2),(0,0,255),0)

        #Get the convex hull
        hull = cv2.convexHull(main_contour)

        drawing = np.zeros(crop_image.shape,np.uint8)
        cv2.drawContours(drawing,[main_contour],-1,(255,255,255),0)
        cv2.drawContours(drawing,[hull],-1,(255,0,0),0)

        # Find the Convexity defect

        hull = cv2.convexHull(main_contour,returnPoints=False)
        defects = cv2.convexityDefects(main_contour,hull)

        #Using Cosine rule
        count_defect = 0

        for i in range(defects.shape[0]):
            startingPoint,endPoint,farthestPoint,ApproxDist = defects[i,0]
            start= tuple(main_contour[startingPoint][0])
            end = tuple(main_contour[endPoint][0])
            far = tuple(main_contour[farthestPoint][0])

            a = math.sqrt((end[0] - start[0])**2 +(end[1]-start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 +(far[1]-start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 +(end[1]-far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/ (2*b*c))*180)/3.14

            #If angle <= 90 draw the circle
            if (angle<=90):
                count_defect+=1
                cv2.circle(crop_image,far,1,[0,0,255],-1)
            cv2.line(crop_image,start,end,[0,0,255],2)

        if (count_defect <=3 and count_defect >=0):
            cv2.putText(frames,"Please keep hands properly",(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),1)

        elif(count_defect >3 and count_defect <=5):
            cv2.putText(frames,"True",(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),1)

        else:
            pass

        all_image = np.hstack((drawing,crop_image))
        cv2.imshow("Countours",all_image)

    except:
        pass

    cv2.imshow("Gestures",frames)

    if cv2.waitKey(1) == ord('a'):
        break

cam.release()
cv2.destroyAllWindows()




