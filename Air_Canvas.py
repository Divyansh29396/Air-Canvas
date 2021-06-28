# Modules used
import cv2
import Hand_Tracking as ht
import time 
import numpy as np
import os 

path1 = "Images"
folder = os.listdir(path1)

img = []
for i in folder:
     m = cv2.imread(f'{path1}/{i}')
     img.append(m)
color_index = img[1]

image = np.zeros((720,1280,3),np.uint8)
xt = yt = 0
ct = 0
pt = 0
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

line_thickness = 20
eraser_thickness = 40
color = (255,255,0)

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
            if y1 < 125:
                if 0 < x1 <140:
                    color_index = img[0]
                    color = (0,0,255)
                elif 185 < x1 < 320:
                    color_index = img[1]
                    color = (255,0,0)
                elif 375 < x1 < 515:
                    color_index = img[2]
                    color = (0,255,0)
                elif 570 < x1 < 715:
                    color_index = img[3]
                    color = (0,255,255)
                elif 775 < x1 < 925:
                    color_index = img[4]
                    color = (255,0,255)
                elif 1045 < x1 < 1280:
                    color_index = img[5]
                    color = (0,0,0)
                    
            cv2.rectangle(frame,(x1,y1-20),(x2,y2),color,-1)
            
        if fingers[1] and fingers[2] == False:
            cv2.circle(frame,(x1,y1),15,color,-1)
            
            if xt == 0 and yt ==0:
                xt = x1
                yt = y1
                
            if color == (0,0,0):
                cv2.line(frame,(xt,yt),(x1,y1),color,eraser_thickness)
                cv2.line(image,(xt,yt),(x1,y1),color,eraser_thickness)
                
            cv2.line(frame,(xt,yt),(x1,y1),color,line_thickness)
            cv2.line(image,(xt,yt),(x1,y1),color,line_thickness)
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
    
    frame[0:125,0:1280] = color_index
    #cv2.imshow("inv",inv)
    #cv2.imshow("Image",image)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
