import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
while True:
    try: 
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
    
        kernel = np.ones((3, 3), np.uint8)
    
        roi = frame[100:300, 100:300]
    
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
        #range of skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
        #extract skin colur 
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
        #extrapolate hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4)
    
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
    
        o, contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        #find contour of max area(hand)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
    
        #approximating the contours
        epsilon = 0.0005*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
    
        hull = cv2.convexHull(cnt)
    
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
    
        #percentage of area not covered by hand in convex hull
        arearatio = ((areahull-areacnt)/areacnt)*100
    
        #defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
    
        l = 0
    
        #finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)
    
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            d = (2*ar)/a
    
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)
            cv2.line(roi, start, end, [0, 255, 0], 2)
    
        l += 1
    
        #corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if areacnt < 2000:
                cv2.putText(frame, 'Put hand in the box', (0, 50),
                            font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, '0', (0, 50), font, 2,
                                (0, 0, 255), 3, cv2.LINE_AA)
                elif arearatio < 17.5:
                    cv2.putText(frame, 'Good Job', (0, 50), font,
                                2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame, '1', (0, 50), font, 2,
                                (0, 0, 255), 3, cv2.LINE_AA)
    
        elif l == 2:
            cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    
        elif l == 3:
        
            if arearatio < 27:
                cv2.putText(frame, '3', (0, 50), font, 2,
                            (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'ok', (0, 50), font, 2,
                            (0, 0, 255), 3, cv2.LINE_AA)
    
        elif l == 4:
            cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    
        elif l == 5:
            cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    
        elif l == 6:
            cv2.putText(frame, 'reposition', (0, 50), font,
                        2, (0, 0, 255), 3, cv2.LINE_AA)
    
        else:
            cv2.putText(frame, 'reposition', (10, 50), font,
                        2, (0, 0, 255), 3, cv2.LINE_AA)
    
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

    except:
        pass

    if cv2.waitKey(75) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
