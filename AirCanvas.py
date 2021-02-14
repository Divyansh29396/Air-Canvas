# Modules used 
import cv2
import numpy as np
from collections import deque

# Function used by trackbars
def empty():
    pass

# Colors i am using
colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorindex = 0

# Creating Window
cv2.namedWindow("Color Detector")

# Creating Trackbars for detecting colours
cv2.createTrackbar("Low Hue","Color Detector",50,180,empty)
cv2.createTrackbar("Low Saturation","Color Detector",62,255,empty)
cv2.createTrackbar("Low Value","Color Detector",60,255,empty)
cv2.createTrackbar("High Hue","Color Detector",85,180,empty)
cv2.createTrackbar("High Saturation","Color Detector",255,255,empty)
cv2.createTrackbar("High Value","Color Detector",255,255,empty)

# Used in morphological functions like erosion, dilation
kernel = np.zeros([3,3],np.uint8)

# Creating image using numpy
image = np.zeros((480,640,3),np.uint8) + 255

# Creating rectangles
cv2.rectangle(image,(20,5),(120,70),(255,0,0),-1)
cv2.rectangle(image,(140,5),(240,70),(0,255,0),-1)
cv2.rectangle(image,(260,5),(360,70),(0,0,255),-1)
cv2.rectangle(image,(380,5),(490,70),(0,255,255),-1)
cv2.rectangle(image,(510,5),(630,70),(255,255,255),-1)

# Writing text in image
cv2.putText(image,"Blue",(35,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
cv2.putText(image,"Green",(155,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
cv2.putText(image,"Red",(275,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
cv2.putText(image,"Yellow",(395,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
cv2.putText(image,"ClearAll",(515,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)

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

# Looping 
while cap.isOpened:
    # Reading frames 
    ret,frame = cap.read()
    # Fliping the frames horizontally
    flip = cv2.flip(frame,1)
     # Changing the color from bgr to hsv 
    hsv = cv2.cvtColor(flip,cv2.COLOR_BGR2HSV)
    
    frame = cv2.rectangle(flip,(20,5),(120,70),(255,0,0),-1)
    frame = cv2.rectangle(frame,(140,5),(240,70),(0,255,0),-1)
    frame = cv2.rectangle(frame,(260,5),(360,70),(0,0,255),-1)
    frame = cv2.rectangle(frame,(380,5),(490,70),(0,255,255),-1)
    frame = cv2.rectangle(frame,(510,5),(630,70),(255,255,255),-1)
    
    frame = cv2.putText(frame,"Blue",(35,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    frame = cv2.putText(frame,"Green",(155,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    frame = cv2.putText(frame,"Red",(275,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    frame = cv2.putText(frame,"Yellow",(395,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    frame = cv2.putText(frame,"ClearAll",(515,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    
   
    
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

    # Finding Contours
    contours,heirarchy = cv2.findContours(dilate.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    # If contpours are formed
    if len(contours) > 0:
        # Find largest contour
        contour = sorted(contours,key=cv2.contourArea,reverse=True)[0]
        # Gets radius and coordinates of circle around detected contour
        ((x,y),radius) = cv2.minEnclosingCircle(contour)
        # Draws a circle
        cv2.circle(frame,(int(x),int(y)),int(radius),(255,255,0),2)
        # Calculate centre of contour
        m = cv2.moments(contour)
        center = (int(m['m10']/m['m00']),int(m['m01']/m['m00']))
        
        # If user cicks on any button
        if 5 <= center[1] <= 70:
            
            if 20 <= center[0] <= 120:
                colorindex = 0  # Blue  
            elif 140 <= center[0] <= 240:
                colorindex = 1  # Green
            elif 260 <= center[0] <= 360:
                colorindex = 2  # Red
            elif 380 <= center[0] <= 490:
                colorindex = 3  # Yellow
            elif 510 <= center[0] <= 630:  # All Clear
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
                
    else:
        bpoints.append(deque(maxlen=512))
        b_index += 1
        gpoints.append(deque(maxlen=512))
        g_index += 1
        rpoints.append(deque(maxlen=512))
        r_index += 1
        ypoints.append(deque(maxlen=512))
        y_index += 1
        
    # Drwaing lines of different colours on canvas
    points = [bpoints,gpoints,rpoints,ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame,points[i][j][k-1],points[i][j][k],colors[i],2)
                cv2.line(image,points[i][j][k-1],points[i][j][k],colors[i],2)
                
    # Showing the captured frame 
    cv2.imshow("frame",frame)
    cv2.imshow("image",image)
    cv2.imshow("mask",mask)
    
    # Breaking the loop when q is pressed
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    
# Releasing captured frame    
cap.release()
# Destroying all the created windows 
cv2.destroyAllWindows()