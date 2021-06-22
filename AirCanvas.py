# Modules used 

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function used by trackbars
def empty():
    pass

# Colors i am using
colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorindex = 0

cv2.namedWindow("Color Detector")

# Creating Trackbars for detecting colours
cv2.createTrackbar("Low Hue","Color Detector",115,180,empty)
cv2.createTrackbar("Low Saturation","Color Detector",135,255,empty)
cv2.createTrackbar("Low Value","Color Detector",105,255,empty)
cv2.createTrackbar("High Hue","Color Detector",177,180,empty)
cv2.createTrackbar("High Saturation","Color Detector",255,255,empty)
cv2.createTrackbar("High Value","Color Detector",255,255,empty)

# Used in morphological functions like erosion, dilation
kernel = np.zeros([3,3],np.uint8)

# Indexes used for marking points for different colour array  
b_index = 0
g_index = 0
r_index = 0
y_index = 0

# Creating arrays for handling different colours
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Starting device camera for live video
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

# Looping 
    while cap.isOpened:
    # Reading frames 
        ret,frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
            continue

        # Fliping the frames horizontally
        flip = cv2.flip(frame,1)
        # Changing the color from bgr to hsv 
        frame = cv2.cvtColor(flip,cv2.COLOR_BGR2RGB)
        # Mark image as non writeable
        frame.flags.writeable = False
        results = hands.process(frame)
        
        # Draw hand annotations on image
        frame.flags.writeable = True
        frame1 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            # Looping all the 21 landmark points
            for hand_landmarks in results.multi_hand_landmarks:
                # To get id and postion of each landmark
                for id, lm in enumerate(hand_landmarks.landmark):
                    print(id, lm)
                    h,w,c = frame1.shape
                    # x and y coordinates of position of landmark
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    print(id,cx,cy)
                    if id == 8:
                        # Drawing circle on landmark 
                        cv2.circle(frame1,(cx,cy),20,(255,0,0),-1)
                
                # Drawing all the hand landmark points
                mp_drawing.draw_landmarks(
                    frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Changing color from BGR to HSV
        hsv = cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)
        
        # Drawing rectangles for different colors
        frame1 = cv2.rectangle(frame1,(20,5),(120,70),(255,0,0),-1)
        frame1 = cv2.rectangle(frame1,(140,5),(240,70),(0,255,0),-1)
        frame1 = cv2.rectangle(frame1,(260,5),(360,70),(0,0,255),-1)
        frame1 = cv2.rectangle(frame1,(380,5),(490,70),(0,255,255),-1)
        frame1 = cv2.rectangle(frame1,(510,5),(630,70),(255,255,255),-1)
        
        # Putting text
        frame1 = cv2.putText(frame1,"Blue",(35,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        frame1 = cv2.putText(frame1,"Green",(155,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        frame1 = cv2.putText(frame1,"Red",(275,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        frame1 = cv2.putText(frame1,"Yellow",(395,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        frame1 = cv2.putText(frame1,"ClearAll",(515,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
      
        # Getting trackbar positions 
        low_hue = cv2.getTrackbarPos("Low Hue","Color Detector")
        low_saturation = cv2.getTrackbarPos("Low Saturation","Color Detector")
        low_value = cv2.getTrackbarPos("Low Value","Color Detector")
        high_hue = cv2.getTrackbarPos("High Hue","Color Detector")
        high_saturation = cv2.getTrackbarPos("High Saturation","Color Detector")
        high_value = cv2.getTrackbarPos("High Value","Color Detector")
        
        # Creating lower hsv value and higher hsv value
        low_hsv = np.array([low_hue,low_saturation,low_value])
        high_hsv = np.array([high_hue,high_saturation,high_value])
        
        # Masking a colour
        mask = cv2.inRange(hsv,low_hsv,high_hsv)
        # Erosion for removing the white noise
        erosion = cv2.erode(mask,kernel,1)
        # Dilation for increasing white region
        dilate = cv2.dilate(erosion,kernel,1)
        
        contours,heirarchy = cv2.findContours(dilate.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center = None
        
        if len(contours) > 0:
            # Find largest contour
            contour = sorted(contours,key=cv2.contourArea,reverse=True)[0]
            # Gets radius and coordinates of circle around detected contour
            ((x,y),radius) = cv2.minEnclosingCircle(contour)
            # Draws a circle
            cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
            # Calculate centre of contour
            m = cv2.moments(contour)
            center = (int(m['m10']/m['m00']),int(m['m01']/m['m00']))
            
            if 5 <= center[1] <= 70:
                if 20 <= center[0] <= 120:
                    colorindex = 0  # Blue  
                elif 140 <= center[0] <= 240:
                    colorindex = 1  # Green
                elif 260 <= center[0] <= 360:
                    colorindex = 2  # Red
                elif 380 <= center[0] <= 490:
                    colorindex = 3  # Yellow
                elif 510 <= center[0] <= 630:
                      b_index = 0
                      g_index = 0
                      r_index = 0
                      y_index = 0
                
                      bpoints = [deque(maxlen=512)]
                      gpoints = [deque(maxlen=512)]
                      rpoints = [deque(maxlen=512)]
                      ypoints = [deque(maxlen=1024)]
                
            else:
                if colorindex == 0:
                    bpoints[b_index].appendleft(center)
                elif colorindex == 1:
                    gpoints[g_index].appendleft(center)
                elif colorindex == 2:
                    rpoints[r_index].appendleft(center)
                elif colorindex == 3:
                    ypoints[y_index].appendleft(center)
       
        points = [bpoints,gpoints,rpoints,ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k-1] is None or points[i][j][k] is None:
                        continue
                    elif (points[i][j][k-1] ==(0,0,0) ) and (points[i][j][k]== (0,0,0)):
                        continue
                    cv2.line(frame1,points[i][j][k-1],points[i][j][k],colors[i],2)
                    #cv2.line(image,points[i][j][k-1],points[i][j][k],colors[i],2)
            
        # Showing the captured frame 
        cv2.imshow("frame",frame1)
        #cv2.imshow("mask",mask)
    
        # Breaking the loop when q is pressed
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    
# Releasing captured frame    
cap.release()
# Destroying all the created windows 
cv2.destroyAllWindows()
