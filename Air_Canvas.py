# Modules used
import cv2
import Hand_Tracking as ht
import time 
import numpy as np

image = np.zeros((720,1280,3),np.uint8)
xt = yt = 0
ct = 0
pt = 0
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detect = ht.handTracking()

while cap.isOpened:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    
    frame = detect.find_number_of_hands(frame)
    position = detect.find_position_of_hands(frame)
    #print(position)
    if len(position) != 0:
        x1,y1 = position[8][1:]
        x2,y2 = position[12][1:]
        
        fingers = detect.find_number_of_fingers()
        #print(fingers)
        if fingers[1] and fingers[2]:
            xt = yt = 0
            cv2.circle(frame,(x2,y2),10,(0,0,0),-1)
            
        if fingers[1] and fingers[2] == False:
            cv2.circle(frame,(x1,y1),15,(255,0,0),-1)
            
            if xt == 0 and yt ==0:
                xt = x1
                yt = y1
            cv2.line(frame,(xt,yt),(x1,y1),(255,255,0),15)
            cv2.line(image,(xt,yt),(x1,y1),(255,255,0),15)
            xt = x1
            yt = y1
    
    ct = time.time()
    fps = 1 / (ct - pt)
    pt = ct
    cv2.putText(frame,str(int(fps)),(30,60),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Masking the image
    _,mask = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
   
    # Bitwise operations 
    frame = cv2.bitwise_and(frame,inv)
    frame = cv2.bitwise_or(frame,image)
    
    #cv2.imshow("inv",inv)
    #cv2.imshow("Image",image)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()