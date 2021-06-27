# Modules Used
import cv2
import mediapipe as mp
import time

class handTracking():
    def __init__(self,static_image_mode=False,
                 max_number_of_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.max_number_of_hands = max_number_of_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mphands = mp.solutions.hands
        self.mpdraw = mp.solutions.drawing_utils
        self.hands = self.mphands.Hands(self.static_image_mode,
                                        self.max_number_of_hands,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)   
        self.fingers_point = [8,12,16,20]
    
    # Method to find number of hands 
    def find_number_of_hands(self,frame,to_draw=True):
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        
        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                # hand_landmark are 21 points. so we need conection too-->mpHands.HAND_CONNECTIONS
                if to_draw:
                    # Drawing the landmarks
                    self.mpdraw.draw_landmarks(frame,hand_landmark,self.mphands.HAND_CONNECTIONS)
        return frame
    
    # Method to find position of landmarks
    def find_position_of_hands(self,frame,hand_no=0):
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            hand_landmark = self.results.multi_hand_landmarks[0]
            for id,lm in enumerate(hand_landmark.landmark):
                w,h,c = frame.shape
                #lm = x,y cordinate of each landmark in float numbers. lm.x, lm.y methods
                cx = int(lm.x*h)
                cy = int(lm.y*w)
                self.lmlist.append([id,cx,cy])
        return self.lmlist
    
    # Find number of fingers
    def find_number_of_fingers(self):
        fingers = []
        if len(self.lmlist) != 0:
            if self.lmlist[4][1] < self.lmlist[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
                
            for i in self.fingers_point:
                if self.lmlist[i][2] < self.lmlist[i-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

def main():
    # Capturing live video
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    detect = handTracking()
    ct = 0
    pt = 0
    
    while cap.isOpened:
        # Reading captured frames
        ret,frame = cap.read()
        # Fliiping the frames 
        frame = cv2.flip(frame,1)
        if not ret:
            continue
        
        frame = detect.find_number_of_hands(frame)
        position = detect.find_position_of_hands(frame)
        if len(position) != 0:
            x1,y1 = position[8][1:]
            x2,y2 = position[12][1:]
        
            fingers = detect.find_number_of_fingers()
            print(fingers)
        
        ct = time.time()
        fps = 1 / (ct - pt)
        pt = ct
        cv2.putText(frame,str(int(fps)),(30,60),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        
        # Showing the captured frame
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
        
if __name__ == "__main__":
    main()